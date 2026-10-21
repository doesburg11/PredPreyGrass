"""Train one condition of the foraging-swarm comparison (README, "Stage 2").

    python predpreygrass/non_evolutionary/continuous_space/foraging_swarms/launchers/launch.py <condition> <seed>

Conditions (prey manipulated first; predators keep full information):
    A   own species seen with positions and velocities (as BE-04 and BE-05)
    B   prey see other prey's positions only (velocities zeroed)
    C   prey see no other prey (own-species slots empty)
    BP  predators see other predators' positions only
    CP  predators see no other predators
    S1  shared carcass: a catch's energy split among predators within 96 units
    S2  shared carcass and slower predators (4.5 instead of 5)
    K   warm start (seed 1: BE-04, 2: BE-05 at iteration 2,000), then 1,000
        iterations with large prey (energy >= 6) needing a second predator
        within 128 units
    W   the same warm start, 1,000 iterations in the unchanged world (control)

Everything else is config/config_env.py (the quirks-fixed milestone world)
and config/config_ppo.py, with 14 simulation workers so that two runs can
train side by side, as BE-04 and BE-05 did. ITERATIONS is 2,000, not
BE-04/05's 3,000: condition A is compared at their saved iteration-2,000
checkpoints (README, "Stage 2"). Long runs (about 9.5 hours):
start them with systemd-run --user so they survive the session, e.g.

    systemd-run --user --unit=sw-B-1 --working-directory=$PWD \\
        .conda/bin/python predpreygrass/.../launchers/launch.py B 1
"""
import sys
from pathlib import Path

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import train
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.config.config_ppo import RUNS_DIR

ITERATIONS = 2000

CONDITIONS = {
    "A": {},
    "B": {"prey_observe_neighbor_movement": False},
    "C": {"prey_observe_neighbors": False},
    "BP": {"predator_observe_neighbor_movement": False},
    "CP": {"predator_observe_neighbors": False},
    # Shared carcass (README, "Shared carcass"): a catch's energy split among
    # the catcher and every predator within 96 units; S2 also makes solo
    # catches harder with slower predators (top speed 4.5 instead of 5;
    # prey 6.0).
    "S1": {"energy_catch_share_radius": 96.0},
    "S2": {"energy_catch_share_radius": 96.0, "predator_max_velocity": 4.5},
    # Large prey (README, "Large prey"), warm-started (WARM_START): prey with
    # energy >= 6 need a second predator within 128 units to be caught, the
    # energy shared with it. W: the same warm start in the unchanged world
    # (control). Stricter settings let even warm-started predators die out.
    "K": {"capture_assist_radius": 128.0, "capture_assist_threshold": 6.0},
    "W": {},
}

# Conditions that start from trained policies (weights only: optimizer,
# iteration count and outputs start fresh) and train WARM_ITERATIONS more:
# seed 1 from BE-04, seed 2 from BE-05, both at iteration 2,000.
_BE_RUNS = Path(__file__).resolve().parents[2] / "base_environment_continuous" / "runs"
WARM_START = {
    1: _BE_RUNS / "base_environment_continuous_2026-10-14_21-35-19_937331" / "checkpoint" / "iter_002000",
    2: _BE_RUNS / "base_environment_continuous_2026-10-14_21-35-40_020808" / "checkpoint" / "iter_002000",
}
WARM_CONDITIONS = ("K", "W")
WARM_ITERATIONS = 1000

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in CONDITIONS:
        raise SystemExit(f"usage: launch.py <{'|'.join(CONDITIONS)}> <seed>")
    condition, seed = sys.argv[1], int(sys.argv[2])
    train.config_env.update(CONDITIONS[condition])
    run = f"{{run_name}}_{condition}_seed{seed}"
    warm = condition in WARM_CONDITIONS
    if warm and seed not in WARM_START:
        raise SystemExit(f"{condition} needs seed 1 or 2 (the warm-start checkpoint)")
    train.config_ppo.update(
        iterations=WARM_ITERATIONS if warm else ITERATIONS,
        init_checkpoint=str(WARM_START[seed]) if warm else None,
        frozen_policies=[],
        num_env_runners=14,
        seed=seed,
        checkpoint_dir=str(RUNS_DIR / run / "checkpoint"),
        tensorboard_dir=str(RUNS_DIR / run / "tensorboard"),
    )
    sys.argv = sys.argv[:1]  # train.py takes no command-line arguments
    train.main()
