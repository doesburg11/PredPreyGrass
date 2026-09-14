"""Does a REAL evolving population maintain two competing reward-genome strategies at
once, or does one outcompete the other? Follow-up to positive_control.py's fitness
result: two hand-picked FIXED genomes (`avoider`, rewarded for fleeing predators, vs.
`anti_adaptive`, rewarded for approaching them) showed real but opposite-pointing
fitness advantages when each was run in its OWN segregated population (avoider:
larger standing population; anti_adaptive: far more total offspring and deeper
lineages, via predation-driven density release). That result compared two SEPARATE,
non-competing populations -- it does not show what happens when both strategies
compete for the same food/space within ONE mixed population, where an individual's
fitness depends on relative, not absolute, success.

This script runs REAL evolution (mutation on, standard neutral random founder init --
no fixed genomes, no fixed-genome comparison) at the learning rate already confirmed
to make reward-genome content matter for behavior and fitness (see lr_sweep.py/
positive_control.py: 20x default), for many generations, and tests whether the
population's `predator_proximity` eval_weight (the single axis that separated
avoider from anti_adaptive) becomes and STAYS bimodal -- a genuine, persisting
two-strategy polymorphism -- rather than staying unimodal (one strategy wins, or the
population just diffuses without splitting).

Bimodality is tested formally at each of several checkpoints through each run: fit a
1-component and a 2-component Gaussian mixture to the live population's
predator_proximity weights via sklearn, compare by BIC (lower is better; the standard
Kass & Raftery scale treats DeltaBIC > 10 as "very strong" evidence for the better
model) -- the same "fit competing models, compare by information criterion" logic as
model_selection.py's Hunt (2006) drift-vs-selection test used earlier in this trial's
investigation, applied here to population structure instead of a single trait's
generation-by-generation trajectory.

Rigor notes, added after a Codex review of the first version (mirroring the same
review-then-fix pattern that caught a real false-positive in lr_sweep.py earlier in
this investigation -- worth repeating here rather than trusting the first pass):

  - NULL-CALIBRATED false-positive rate: at every checkpoint, a synthetic sample of
    the SAME SIZE drawn from a single Gaussian matched to the real sample's mean/std
    is fit with the identical 1-vs-2-component procedure, to measure how often even a
    KNOWN-unimodal population would spuriously cross delta_BIC > 10 at this exact
    sample size -- the same "simulate the null, measure detection rate" logic as the
    earlier Hunt-test power check.
  - n_init=10 (not sklearn's default of 1) for the 2-component fit, so a result isn't
    just whichever local optimum one random initialization happened to land in.
  - Persistence, not a single snapshot: a seed only counts as showing a maintained
    polymorphism if delta_BIC > 10 holds at ALL of its last 3 checkpoints, not just
    the final one or a lucky earlier one.
  - Denominator tracked honestly: populations with <10 live agents (small/near-extinct)
    are excluded from a checkpoint's BIC fit as before, but the report shows how many
    seeds were actually eligible at each step, not just a fraction among survivors that
    could quietly mean "1 of 1."
  - Per-seed exceptions (corrupt checkpoint, degenerate fit) are caught and reported,
    not left to crash the whole batch.

Usage:
    python -m predpreygrass.evolutionary.eco_evolutionary_erl_flagship.polymorphism_check \\
        --steps 20000 --seeds 10 --checkpoint-every 2000
"""

import argparse
import math
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.checkpoint import load_checkpoint
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import FEATURE_NAMES
from predpreygrass.global_config import ERL_RESULTS_DIR

PRED_PROX_IDX = FEATURE_NAMES.index("predator_proximity")
MIN_LIVE_AGENTS = 10
N_INIT_2COMP = 10
DELTA_BIC_THRESHOLD = 10.0


def _fit_1_vs_2(x: np.ndarray, seed: int) -> tuple[float, float, GaussianMixture]:
    gmm1 = GaussianMixture(n_components=1, random_state=seed).fit(x)
    gmm2 = GaussianMixture(n_components=2, random_state=seed, n_init=N_INIT_2COMP).fit(x)
    return gmm1.bic(x), gmm2.bic(x), gmm2


def bic_bimodality(values: np.ndarray, seed: int, rng: np.random.Generator) -> dict:
    """Fit 1- and 2-component 1D Gaussian mixtures to `values`; return BIC for each,
    the delta, mode info (if 2-component wins), and a null-calibrated false-positive
    check: the same procedure applied to a matched-size, matched-mean/std synthetic
    unimodal sample."""
    x = values.reshape(-1, 1)
    bic1, bic2, gmm2 = _fit_1_vs_2(x, seed)
    result = {"n": len(values), "bic1": bic1, "bic2": bic2, "delta_bic": bic1 - bic2}
    if result["delta_bic"] > 0:
        order = np.argsort(gmm2.means_.ravel())
        result["mode_means"] = gmm2.means_.ravel()[order].tolist()
        result["mode_weights"] = gmm2.weights_[order].tolist()

    null_x = rng.normal(values.mean(), values.std(), size=len(values)).reshape(-1, 1)
    null_bic1, null_bic2, _ = _fit_1_vs_2(null_x, seed)
    result["null_delta_bic"] = null_bic1 - null_bic2
    return result


def run_one(seed: int, steps: int, checkpoint_every: int, lr_multiplier: float, out_root: Path) -> dict:
    out_dir = out_root / f"seed_{seed}"
    cmd = [
        sys.executable, "-m",
        "predpreygrass.evolutionary.eco_evolutionary_erl_flagship.run_trial13_simulation",
        "--steps", str(steps),
        "--seed", str(seed),
        "--log-every", str(max(checkpoint_every, 2000)),
        "--checkpoint-every", str(checkpoint_every),
        "--out-dir", str(out_dir),
        "--lr-multiplier", str(lr_multiplier),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return {"seed": seed, "ok": False, "stderr_tail": f"subprocess.run raised: {e}"}
    if result.returncode != 0:
        return {"seed": seed, "ok": False, "stderr_tail": result.stderr[-2000:]}

    rng = np.random.default_rng(seed)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_step_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    per_checkpoint, n_ineligible = [], 0
    for ckpt_path in checkpoints:
        step = int(ckpt_path.stem.split("_")[-1])
        try:
            payload = load_checkpoint(ckpt_path)
            registry = payload["registry"]
            if len(registry) < MIN_LIVE_AGENTS:
                n_ineligible += 1
                continue
            pred_prox_weights = np.array([s.genome.eval_weights[PRED_PROX_IDX] for s in registry.values()])
            bic_result = bic_bimodality(pred_prox_weights, seed, rng)
            bic_result["step"] = step
            bic_result["mean_generation"] = float(np.mean([s.generation for s in registry.values()]))
            per_checkpoint.append(bic_result)
        except Exception:
            return {
                "seed": seed, "ok": False,
                "stderr_tail": f"Exception analyzing {ckpt_path}:\n{traceback.format_exc()}",
            }

    sustained = False
    if len(per_checkpoint) >= 3:
        sustained = all(c["delta_bic"] > DELTA_BIC_THRESHOLD for c in per_checkpoint[-3:])

    return {
        "seed": seed, "ok": True, "checkpoints": per_checkpoint,
        "n_checkpoints_total": len(checkpoints), "n_ineligible": n_ineligible,
        "sustained_bimodal_last3": sustained,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--lr-multiplier", type=float, default=20.0,
                         help="Default matches the multiplier already confirmed (lr_sweep.py, "
                              "positive_control.py) to make reward-genome content reach behavior "
                              "and fitness at detectable strength.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.seeds < 1 or args.steps < 1 or args.checkpoint_every < 1:
        raise ValueError("--seeds, --steps, and --checkpoint-every must all be >= 1.")
    if not (math.isfinite(args.lr_multiplier) and args.lr_multiplier >= 0):
        raise ValueError(f"--lr-multiplier must be finite and >= 0, got {args.lr_multiplier}.")

    import os
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_root = Path(args.out_dir) if args.out_dir else ERL_RESULTS_DIR / f"ERL_FLAGSHIP_polymorphism_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Launching {args.seeds} seeds, {args.steps} steps each, lr_multiplier={args.lr_multiplier}, "
          f"checkpoint every {args.checkpoint_every} steps, {workers} parallel workers.")
    print(f"Output: {out_root}")

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, seed, args.steps, args.checkpoint_every, args.lr_multiplier, out_root): seed
            for seed in range(1, args.seeds + 1)
        }
        done = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                row = fut.result()
            except Exception:
                row = {"seed": seed, "ok": False, "stderr_tail": traceback.format_exc()}
            results.append(row)
            done += 1
            print(f"[{done}/{args.seeds}] seed={seed}: {'OK' if row['ok'] else 'FAILED'}")

    elapsed = time.time() - start
    print(f"\nAll runs finished in {elapsed:.0f}s.\n")

    ok_results = [r for r in results if r["ok"]]
    all_steps = sorted({c["step"] for r in ok_results for c in r["checkpoints"]})

    print(f"{'step':<10}{'mean_gen':<12}{'n_eligible':<12}{'frac_bimodal':<14}{'null_fp_rate':<14}"
          f"{'mean_delta_bic':<16}{'mode_means (when bimodal)'}")
    for step in all_steps:
        rows_at_step = [c for r in ok_results for c in r["checkpoints"] if c["step"] == step]
        if not rows_at_step:
            continue
        n = len(rows_at_step)
        bimodal = [c for c in rows_at_step if c["delta_bic"] > DELTA_BIC_THRESHOLD]
        null_bimodal = [c for c in rows_at_step if c["null_delta_bic"] > DELTA_BIC_THRESHOLD]
        frac_bimodal = len(bimodal) / n
        null_fp_rate = len(null_bimodal) / n
        mean_dbic = sum(c["delta_bic"] for c in rows_at_step) / n
        mean_gen = sum(c["mean_generation"] for c in rows_at_step) / n
        mode_str = ""
        if bimodal:
            mode_pairs = [f"({m[0]:+.2f}, {m[1]:+.2f})" for m in (c["mode_means"] for c in bimodal)]
            mode_str = "; ".join(mode_pairs[:5]) + (" ..." if len(mode_pairs) > 5 else "")
        print(f"{step:<10}{mean_gen:<12.1f}{n:<12}{frac_bimodal:<14.2f}{null_fp_rate:<14.2f}"
              f"{mean_dbic:<16.1f}{mode_str}")

    n_sustained = sum(1 for r in ok_results if r.get("sustained_bimodal_last3"))
    print(f"\nSustained polymorphism (delta_BIC > {DELTA_BIC_THRESHOLD:.0f} at ALL of a seed's last 3 "
          f"checkpoints, not just one snapshot): {n_sustained}/{len(ok_results)} seeds.")

    print(
        "\nfrac_bimodal = fraction of ELIGIBLE (>=10 live agents) checkpoints at this step where a "
        "2-component Gaussian mixture on live population predator_proximity eval_weights beats a "
        "1-component fit by 'very strong' evidence (Kass & Raftery scale, delta_BIC > 10). "
        "null_fp_rate = the SAME test applied to a synthetic unimodal sample matched to the real "
        "sample's size/mean/std at this checkpoint -- the calibrated false-positive rate; frac_bimodal "
        "should clear null_fp_rate by a wide margin before treating this as a real signal, not just "
        "noise the test is prone to at this population size. avoider's founder genome used "
        "predator_proximity=-2.0, anti_adaptive's used +2.0 -- mode_means near those values, with "
        "frac_bimodal >> null_fp_rate and persisting across checkpoints, would indicate a real, "
        "maintained polymorphism on this specific reward-weight axis (not yet a claim about realized "
        "behavior -- see behavior_diagnostic.py for that separate check, if this signal holds up)."
    )

    failed = [r for r in results if not r["ok"]]
    if failed:
        print(f"\n{len(failed)} run(s) failed. First failure's stderr tail:")
        print(failed[0]["stderr_tail"])


if __name__ == "__main__":
    main()
