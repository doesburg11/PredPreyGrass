"""Pilot (DESIGN_fov_tradeoff.md in eco_evolutionary_continuous_traits, plan
step 2): one run with the field-of-view trade-off and one without, side by
side, from scratch, with CT-05's run settings (3,000 iterations, 10
workers) and this module's environment (config/config_env.py: strike
catching, no energy cost for field of view). Both seeded with seed 1.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov.launchers.pilot ARM [ITERATIONS]

ARM is one of
  tradeoff        monocular distance noise 0.3, monocular field 150
  control         noise 0 (no trade-off; field of view free; the monocular
                  field then has no effect, so this is the control of
                  every arm)
  tradeoff_m120   pilot 2: noise 0.3, monocular field 120 (the predators'
                  150-degree start is no longer fully binocular), seed 1
  tradeoff_m120_seed2  the same with seed 2
Output: runs/pilot_<ARM>/ (or pilot_<ARM>_smoke with ITERATIONS given).
Pre-specified test 1 (divergence; for pilot 2 re-registered in the
design note's addendum of 2026-10-28): the change in the mean field-of-view
gene, start to the mean of the last 300 iterations, is negative for
predators and non-negative for prey in the trade-off run; both rise in the
control.
"""

import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
RUNS = MODULE / "runs"
ARMS = {  # arm: (monocular distance noise, monocular field, seed)
    "tradeoff": (0.3, 150.0, 1),
    "control": (0.0, 150.0, 1),
    "tradeoff_m120": (0.3, 120.0, 1),
    "tradeoff_m120_seed2": (0.3, 120.0, 2),
}
WORKERS = 10


def main():
    arm = sys.argv[1]
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    out = RUNS / (f"pilot_{arm}" + ("" if len(sys.argv) <= 2 else "_smoke"))
    if (out / "checkpoint").exists():
        raise SystemExit(f"{out} already has a checkpoint folder; refusing to overwrite")

    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_fov import train

    noise, field, seed = ARMS[arm]
    train.config_env.update(fov_monocular_distance_noise=noise, fov_monocular_field=field)
    train.config_ppo.update(
        iterations=iterations,
        num_env_runners=WORKERS,
        seed=seed,
        checkpoint_dir=str(out / "checkpoint"),
        tensorboard_dir=str(out / "tensorboard"),
    )
    train.main()


if __name__ == "__main__":
    main()
