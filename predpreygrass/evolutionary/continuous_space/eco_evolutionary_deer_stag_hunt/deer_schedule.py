"""Replayed deer (DESIGN_2.md, RESULTS.md section 2): the deer of a
deer_stag_hunt run, in the order they were trained.

Training iteration t (1-based) collects its samples with the deer of the
source run's checkpoint D(t) (source_iteration):

    D(t) = iteration 10                for t = 1 ... 19
    D(t) = iteration 10 * floor(t/10)  for t = 20 ... 999
    D(1000) = the final checkpoint (iteration 1000)

So the deer change before sampling iterations 20, 30, ..., 990 and 1000
(switch_iterations). Before training, the deer module of every checkpoint a
run will use is copied into <run>/deer_schedule/iter_<n>/ (module_state.pkl
and the RLlib metadata next to it), with manifest.json: per source
iteration, its source path and the SHA-256 of module_state.pkl. Each source
must be a deer_stag_hunt run of the touch rule whose world equals this
run's (genes and deer_stag_hunt's gradual-grass settings aside, with
instant grass there), with RLlib's default deer network.
"""

import hashlib
import json
import pickle
import shutil
from pathlib import Path

SCHEDULE = "deer_schedule"
MANIFEST = "manifest.json"
FIRST, LAST, EVERY = 10, 1000, 10
# deer_stag_hunt settings without a counterpart here, and their required values.
SOURCE_ONLY = {"grass_regrowth": "instant"}
# Settings older deer_stag_hunt runs saved without (their values then).
SOURCE_LEGACY = {"grass_regrowth": "instant", "deer_young_solo": "graded"}
SOURCE_ONLY_FREE = {"grass_growth_rate", "grass_root_reserve", "grass_bite", "grass_stubble", "grass_dieback_delay"}


def source_iteration(t):
    """The source checkpoint's iteration whose deer act in training
    iteration t (1-based); D(t) above."""
    if isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= LAST:
        raise ValueError(f"training iteration must be an integer in [1, {LAST}], not {t!r}")
    if t < 2 * EVERY:
        return FIRST
    return EVERY * (t // EVERY)


def switch_iterations(last):
    """The training iterations (up to last) before whose sampling the deer
    change."""
    return [t for t in range(2, last + 1) if source_iteration(t) != source_iteration(t - 1)]


def source_iterations(last):
    """Every source iteration a run of `last` training iterations uses."""
    return sorted({source_iteration(t) for t in range(1, last + 1)})


def source_checkpoint(source_run, iteration):
    """A deer_stag_hunt run's checkpoint at a source iteration (the final one
    at LAST)."""
    checkpoint = Path(source_run) / "checkpoint"
    return checkpoint if iteration == LAST else checkpoint / f"iter_{iteration:06d}"


def module_dir(checkpoint):
    return Path(checkpoint) / "learner_group" / "learner" / "rl_module" / "deer_policy"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_source(source_run, env_config):
    """Refuse a source run whose world is not this run's (module docstring);
    returns its saved settings."""
    settings_path = Path(source_run) / "source_code" / "settings.json"
    if not settings_path.is_file():
        raise ValueError(f"{source_run}: no source_code/settings.json")
    settings = json.loads(settings_path.read_text())
    source_env, source_ppo = {**SOURCE_LEGACY, **settings["config_env"]}, settings["config_ppo"]
    if source_env.get("deer_catch_rule") != "touch":
        raise ValueError(f"{source_run}: deer_catch_rule {source_env.get('deer_catch_rule')!r}, not 'touch'")
    if source_ppo.get("vf_share_layers", False) is not False:
        raise ValueError(f"{source_run}: the deer network shares layers; expected RLlib's default with vf_share_layers False")
    if source_ppo.get("iterations") != LAST:
        raise ValueError(f"{source_run}: trained {source_ppo.get('iterations')} iterations, not {LAST}")
    problems = []
    for key, value in SOURCE_ONLY.items():
        if source_env.get(key) != value:
            problems.append(f"{key}: {source_env.get(key)!r} there, required {value!r}")
    ours = {k: v for k, v in env_config.items() if not k.startswith("gene_")}
    theirs = {k: v for k, v in source_env.items() if k not in SOURCE_ONLY and k not in SOURCE_ONLY_FREE}
    for key in sorted(set(ours) | set(theirs)):
        if ours.get(key, "<missing>") != theirs.get(key, "<missing>"):
            problems.append(f"{key}: {theirs.get(key, '<missing>')!r} there, {ours.get(key, '<missing>')!r} here")
    if problems:
        raise ValueError(f"{source_run}: not this run's world:\n  " + "\n  ".join(problems))
    return settings


def copy_schedule(source_run, run_dir, last, env_config):
    """Copy every deer module a run of `last` iterations uses into
    <run_dir>/deer_schedule/, with the manifest; returns the manifest."""
    check_source(source_run, env_config)
    target = Path(run_dir) / SCHEDULE
    if target.exists():
        raise ValueError(f"{target} exists")
    manifest = {"source_run": str(Path(source_run).resolve()), "last_training_iteration": last, "deer": {}}
    for iteration in source_iterations(last):
        source = module_dir(source_checkpoint(source_run, iteration))
        if not (source / "module_state.pkl").is_file():
            raise ValueError(f"{source}: no module_state.pkl")
        destination = target / f"iter_{iteration:06d}"
        shutil.copytree(source, destination)
        digest = sha256(destination / "module_state.pkl")
        if digest != sha256(source / "module_state.pkl"):
            raise RuntimeError(f"{destination}: copy differs from its source")
        manifest["deer"][str(iteration)] = {"source": str(source.resolve()), "sha256": digest}
    (target / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def manifest(run_dir):
    path = Path(run_dir) / SCHEDULE / MANIFEST
    return json.loads(path.read_text()) if path.is_file() else None


def deer_path(run_dir, iteration):
    """The copied deer module of a source iteration, checked against the
    manifest's hash."""
    entry = manifest(run_dir)["deer"].get(str(iteration))
    if entry is None:
        raise ValueError(f"{run_dir}: no deer of source iteration {iteration} in its schedule")
    path = Path(run_dir) / SCHEDULE / f"iter_{iteration:06d}"
    if sha256(path / "module_state.pkl") != entry["sha256"]:
        raise RuntimeError(f"{path}: module_state.pkl differs from the manifest")
    return path


def load_state(path):
    with open(Path(path) / "module_state.pkl", "rb") as f:
        return pickle.load(f)


ACTIVE = "deer.json"


def write_active(checkpoint, iteration, run_dir):
    """Record next to a checkpoint which deer acted in its last iteration."""
    entry = manifest(run_dir)["deer"][str(iteration)]
    Path(checkpoint).mkdir(parents=True, exist_ok=True)
    (Path(checkpoint) / ACTIVE).write_text(json.dumps({"source_iteration": iteration, "sha256": entry["sha256"]}) + "\n")


def active(checkpoint):
    """{source_iteration, sha256} of the deer a checkpoint was saved with,
    or None (section 1's runs: one frozen deer, <run>/deer_policy)."""
    path = Path(checkpoint) / ACTIVE
    return json.loads(path.read_text()) if path.is_file() else None
