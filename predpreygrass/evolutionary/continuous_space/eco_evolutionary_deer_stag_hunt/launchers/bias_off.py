"""RESULTS.md section 3.5: arm B's checkpoints 500, 750 and 1000 evaluated
with the innate bias switched off (beta 0 at evaluation only), everything
else as in section 3.3: the same policy weights, founder gene pool, deer and
episode seeds (evaluate.py 50 episodes, seeds 0-49; coop_fair.py 20).

- own world: results/bias_off/<run>_iter<mark>_{eval,coop}.json
- panel: results/bias_off/<run>_iter<mark>_vs_<deer run>_eval.json for the
  11 other main runs' deer (L and B), with cross_play.py's provenance.

The bias is removed by setting the loaded predator module's gene_beta to 0
after evaluate._init has built it (rl_module.py adds the bias only when
gene_beta is non-zero); the frozen evaluation code is not changed. Each
worker checks that the module is the gene-bias module with the run's
trained beta before switching it off. Finished files with the same
provenance are skipped, so the launcher can be restarted.

--check RUN MARK: the same code path with the bias left on, compared with
section 3.3's own-world result for that checkpoint (must be identical).

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/bias_off.py [--workers 16]
    python .../bias_off.py --check main3_B1 500
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import evaluate
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers import coop_fair

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "bias_off"
MARKS = (500, 750, 1000)
RUNS = {arm: [f"main3_{arm}{k}" for k in range(1, 7)] for arm in ("L", "B")}
EPISODES, COOP_EPISODES = 50, 20


def switch_off(actors, trained_beta):
    """Set the predator module's gene_beta to 0, after checking it is the
    gene-bias module carrying the run's trained beta (non-zero)."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.rl_module import GeneBiasPPOTorchRLModule

    module = actors["predator"].module
    if not isinstance(module, GeneBiasPPOTorchRLModule):
        raise TypeError(f"predator module is {type(module).__name__}, not GeneBiasPPOTorchRLModule")
    if not trained_beta or module.gene_beta != float(trained_beta):
        raise ValueError(f"predator module beta {module.gene_beta}, run's trained beta {trained_beta}")
    module.gene_beta = 0.0


def _init(checkpoint, deer, bias_on):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import run_settings

    evaluate._init(checkpoint, {}, deer)
    if not bias_on:
        switch_off(evaluate._worker["actors"], run_settings(checkpoint)["config_ppo"]["beta"])


def play(checkpoint, deer, workers, bias_on=False):
    """(evaluation summary, rows, coop summary, coop rows) for a checkpoint
    against a deer module directory, bias off unless bias_on."""
    args = (str(checkpoint), str(deer), bias_on)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init, initargs=args) as pool:
        rows = sorted(pool.map(evaluate._play, range(EPISODES)), key=lambda row: row["seed"])
        coop_rows = list(pool.map(coop_fair._count, range(COOP_EPISODES)))
    summary = {"checkpoint": str(checkpoint), "deer": str(deer), "overrides": {}, "beta_at_evaluation": None if bias_on else 0.0, **evaluate.summarise(rows)}
    return summary, rows, coop_fair.summarise(coop_rows), coop_rows


def done(path, expected):
    try:
        data = json.loads(path.read_text())
        return (
            data.get("provenance") == expected
            and data["summary"].get("beta_at_evaluation") == 0.0
            and data["summary"]["episodes"] == EPISODES == len(data["episodes"])
            and sorted(row["seed"] for row in data["episodes"]) == list(range(EPISODES))
            and data["coop"]["episodes"] == COOP_EPISODES == len(data["coop_episodes"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def cell(hunter_run, deer_run, mark, workers):
    """One bias-off evaluation (own world when deer_run == hunter_run)."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.cross_play import check_pair, provenance

    hunters = checkpoint_at(RUNS_DIR / hunter_run, mark)
    deer_checkpoint = checkpoint_at(RUNS_DIR / deer_run, mark)
    deer = checkpoint_deer(deer_checkpoint)
    if deer_run != hunter_run:
        check_pair(hunters, deer_checkpoint)
    expected = dict(provenance(hunters, deer), beta_at_evaluation=0.0)
    name = f"{hunter_run}_iter{mark}" + ("" if deer_run == hunter_run else f"_vs_{deer_run}") + "_eval.json"
    path = OUT / name
    if done(path, expected):
        return
    summary, rows, coop_summary, coop_rows = play(hunters, deer, workers)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"provenance": expected, "hunter_run": hunter_run, "deer_run": deer_run, "mark": mark, "summary": summary, "episodes": rows, "coop": coop_summary, "coop_episodes": coop_rows}, indent=1))
    temporary.replace(path)
    print(f"bias off, iter {mark}: {hunter_run} hunters vs {deer_run} deer: coexisting {summary['coexisting']}/50, adults/episode {summary['adults_caught_per_episode']:.2f}, adult/chance {coop_summary['adult_over_chance']}", flush=True)


def check(run, mark, workers):
    """Bias left on through this code path must reproduce section 3.3's
    own-world evaluation of the checkpoint exactly."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR

    checkpoint = checkpoint_at(RUNS_DIR / run, mark)
    summary, rows, coop_summary, _ = play(checkpoint, checkpoint_deer(checkpoint), workers, bias_on=True)
    old = json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())
    old_coop = json.loads((RESULTS / f"{run}_iter{mark}_coop.json").read_text())["summary"]
    same = old["episodes"] == json.loads(json.dumps(rows)) and old_coop == json.loads(json.dumps(coop_summary))
    print(f"check {run} iter {mark}, bias on: adults/episode {summary['adults_caught_per_episode']:.2f} (section 3.3: {old['summary']['adults_caught_per_episode']:.2f}); identical: {same}")
    if not same:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--check", nargs=2, metavar=("RUN", "MARK"))
    args = parser.parse_args()
    if args.check:
        check(args.check[0], int(args.check[1]), args.workers)
        return
    OUT.mkdir(exist_ok=True)
    everyone = RUNS["L"] + RUNS["B"]
    for mark in MARKS:
        for run in RUNS["B"]:
            cell(run, run, mark, args.workers)
    for mark in MARKS:
        for run in RUNS["B"]:
            for deer_run in everyone:
                if deer_run != run:
                    cell(run, deer_run, mark, args.workers)


if __name__ == "__main__":
    main()
