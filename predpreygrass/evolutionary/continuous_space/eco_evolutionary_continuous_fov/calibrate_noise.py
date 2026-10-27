"""Calibration of the monocular distance noise (DESIGN_fov_tradeoff.md in
eco_evolutionary_continuous_traits, plan step 1): how much does noise sigma
hurt hunting and escaping, before any policy has learned to live with it?

The CT-05 branch point (eco_evolutionary_continuous_traits at iteration
1,500): its predator and prey policies, which never saw noise, and its gene
pools, held fixed (founders drawn from them every episode). Its own settings
otherwise, field-of-view energy cost included, so that only the noise
differs between conditions. Conditions: sigma in SIGMAS, applied to the
predators only, the prey only, or both (sigma 0 is the common baseline).
Episode seeds are common to all conditions, and so are the noise draws:
every monocular observation draws its factor, also for a species whose noise
is switched off (the factor is then not applied).

Per episode: catches, predator and prey decision steps (animals alive,
summed over decisions), births, survivors; per catch whether the prey was
inside the catching predator's binocular overlap ("front") or outside it
("side") at the moment of contact; and per predator decision whether its
nearest prey in view was in front or to the side (the opportunities).

Read-outs: catches per 1,000 predator decisions against sigma for each
target, as a ratio to sigma 0 with a 95% bootstrap interval over episode
seeds (paired across conditions); and catches per 1,000 front and side
opportunities. Predator-only noise measures the cost to hunting, prey-only
noise the cost to escaping (if catches rise). These are end-to-end rates:
noise can change search, pursuit, prey numbers and encounters, not only the
last approach, and a catch is classified where the prey was at contact, not
where it was first seen.

Writes one JSON line per episode to runs/calibrate_noise/<condition>.jsonl
and a summary to runs/calibrate_noise/summary.json.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.calibrate_noise --episodes 200
"""

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parent
TRAITS_RUNS = MODULE.parent / "eco_evolutionary_continuous_traits" / "runs"
BRANCH_POINT = TRAITS_RUNS / "eco_evolutionary_continuous_traits_2026-10-16_20-53-30_161103" / "checkpoint" / "iter_001500"
OUT = MODULE / "runs" / "calibrate_noise"
SIGMAS = (0.15, 0.3, 0.5)
TARGETS = ("predator", "prey", "both")
CHUNK = 20  # episodes per task


def conditions():
    """{name: (sigma, noisy species)}."""
    out = {"sigma0": (0.0, ())}
    for target in TARGETS:
        for sigma in SIGMAS:
            out[f"sigma{sigma}_{target}"] = (sigma, ("predator", "prey") if target == "both" else (target,))
    return out


def make_env(branch_point, sigma, noisy):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import (
        checkpoint_env_config,
        load_gene_pool,
        use_gene_pool,
    )
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.environment import PredPreyGrassContinuous

    class CalibrationEnv(PredPreyGrassContinuous):
        """Noise for the listed species only; catches split by where the
        prey was in the catcher's view."""

        def reset(self, *, seed=None, options=None):
            self.tally = {"front": 0, "side": 0, "front_seen": 0, "side_seen": 0, "predator_steps": 0, "prey_steps": 0}
            return super().reset(seed=seed, options=options)

        def _monocular_factor(self, observer, dx, dy, scaled_distance):
            factor = super()._monocular_factor(observer, dx, dy, scaled_distance)
            return factor if observer.species in noisy else 1.0

        def _zone(self, predator, prey):
            dx, dy = self.torus.offset(predator.position, prey.position)
            overlap = min(max(2 * self.monocular_field - predator.fov, 0.0), predator.fov)
            bearing = abs((-math.degrees(math.atan2(dy, dx)) - predator.orientation + 180) % 360 - 180)
            return "front" if bearing <= overlap / 2 else "side"

        def _move(self, animal, action, catcher, aways, catches):
            if catcher is not None:
                self.tally[self._zone(catcher, animal)] += 1
            return super()._move(animal, action, catcher, aways, catches)

        def step(self, action_dict):
            self.tally["predator_steps"] += len(self.predators)
            self.tally["prey_steps"] += len(self.prey)
            for predator in self.predators:
                seen = [
                    (math.hypot(*self.torus.offset(predator.position, prey.position)), prey)
                    for prey in self.prey
                    if self.torus.in_view(predator, prey.position, predator.view_distance, predator.fov)
                ]
                if seen:
                    self.tally[self._zone(predator, min(seen, key=lambda entry: entry[0])[1]) + "_seen"] += 1
            return super().step(action_dict)

    config = checkpoint_env_config(branch_point)
    config.update(trait_initial_gene_pool=None, fov_monocular_field=150.0, fov_monocular_distance_noise=sigma)
    env = CalibrationEnv(config)
    use_gene_pool(env, load_gene_pool(branch_point))
    env.freeze_gene_pool = True
    return env


def run_chunk(task):
    import torch

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.checkpoints import load_module
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.tournament import sample_actions

    name, branch_point, sigma, noisy, seeds = task
    torch.set_num_threads(1)
    env = make_env(branch_point, sigma, noisy)
    modules = {species: load_module(branch_point, species) for species in ("predator", "prey")}
    records = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        obs, _ = env.reset(seed=seed)
        steps = 0
        while True:
            obs, _, terms, truncs, _ = env.step(sample_actions(modules, obs, generator, env))
            steps += 1
            if terms["__all__"] or truncs["__all__"]:
                break
            obs = {a: o for a, o in obs.items() if not (terms.get(a) or truncs.get(a))}
        records.append(
            {
                "seed": seed,
                "steps": steps,
                **env.tally,
                "catches": env.tally["front"] + env.tally["side"],
                "births": dict(env.births),
                "predators_alive": len(env.predators),
                "prey_alive": len(env.prey),
            }
        )
    return name, records


RATES = {"catch_rate": ("catches", "predator_steps"), "front_rate": ("front", "front_seen"), "side_rate": ("side", "side_seen")}


def summarize(results, boots=2000):
    """Per condition and rate (catches per 1,000 predator decisions, per
    1,000 front and per 1,000 side opportunities): the pooled rate, its ratio
    to sigma 0 and a 95% bootstrap interval of that ratio, resampling episode
    seeds jointly for both conditions; also the share of catches from the
    side and how often each species survived the episode."""
    seeds = sorted({r["seed"] for r in results["sigma0"]})
    if not seeds or any(sorted(r["seed"] for r in records) != seeds for records in results.values()):
        raise ValueError("every condition needs the same episode seeds")
    picks = np.random.default_rng(0).integers(0, len(seeds), size=(boots, len(seeds)))

    def columns(records, rate):
        top, bottom = RATES[rate]
        records = sorted(records, key=lambda r: r["seed"])
        return np.array([r[top] for r in records], dtype=float), np.array([r[bottom] for r in records], dtype=float)

    def pooled(top, bottom):
        return 1000 * top.sum(axis=-1) / bottom.sum(axis=-1)

    out = {}
    for name, records in results.items():
        entry = {"episodes": len(records)}
        for rate in RATES:
            top, bottom = columns(records, rate)
            base_top, base_bottom = columns(results["sigma0"], rate)
            if bottom.sum() == 0 or base_top.sum() == 0:
                entry[rate] = None
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = pooled(top[picks], bottom[picks]) / pooled(base_top[picks], base_bottom[picks])
            entry[rate] = float(pooled(top, bottom))
            entry[f"{rate}_vs_sigma0"] = float(pooled(top, bottom) / pooled(base_top, base_bottom))
            entry[f"{rate}_vs_sigma0_95"] = [float(x) for x in np.nanpercentile(ratios, [2.5, 97.5])]
        catches = sum(r["catches"] for r in records)
        entry["side_share"] = sum(r["side"] for r in records) / catches if catches else None
        entry["predators_survive"] = float(np.mean([r["predators_alive"] > 0 for r in records]))
        entry["prey_survive"] = float(np.mean([r["prey_alive"] > 0 for r in records]))
        out[name] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--branch-point", default=str(BRANCH_POINT), help="CT-05 checkpoint (iteration 1,500)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not (Path(args.branch_point) / "gene_pools.json").is_file():
        parser.error(f"no CT-05 checkpoint at {args.branch_point} (pass --branch-point)")
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = conditions()
    tasks = [
        (name, args.branch_point, sigma, noisy, list(range(start, min(start + CHUNK, args.episodes))))
        for name, (sigma, noisy) in chosen.items()
        for start in range(0, args.episodes, CHUNK)
    ]
    (OUT / "settings.json").write_text(
        json.dumps({"episodes": args.episodes, "branch_point": args.branch_point, "conditions": {k: [s, list(n)] for k, (s, n) in chosen.items()}}, indent=2) + "\n"
    )
    results = {name: [] for name in chosen}
    start, done = time.time(), 0
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        for name, records in pool.map(run_chunk, tasks):
            results[name] += records
            done += 1
            print(f"{done}/{len(tasks)} chunks, {(time.time() - start) / 60:.1f} min", flush=True)
    for name, records in results.items():
        records.sort(key=lambda r: r["seed"])
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = summarize(results)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'condition':22s}" + "".join(f" {rate + ' (vs 0, 95%)':>30s}" for rate in RATES) + f" {'pred ok':>8s} {'prey ok':>8s}")
    for name, s in summary.items():
        cells = ""
        for rate in RATES:
            if s[rate] is None:
                cells += f" {'-':>30s}"
            else:
                lo, hi = s[f"{rate}_vs_sigma0_95"]
                cells += f" {s[rate]:8.2f} ({s[rate + '_vs_sigma0']:.2f}, {lo:.2f}-{hi:.2f})"
        print(f"{name:22s}{cells} {s['predators_survive']:8.2f} {s['prey_survive']:8.2f}")


if __name__ == "__main__":
    main()
