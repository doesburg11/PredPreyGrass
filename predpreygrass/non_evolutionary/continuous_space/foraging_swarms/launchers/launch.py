"""Train one condition of the foraging-swarm comparison (README, "Stage 2").

    python predpreygrass/non_evolutionary/continuous_space/foraging_swarms/launchers/launch.py <condition> <seed>

Conditions (prey manipulated first; predators keep full information):
    A   own species seen with positions and velocities (as BE-04 and BE-05)
    B   prey see other prey's positions only (velocities zeroed)
    C   prey see no other prey (own-species slots empty)
    BP  predators see other predators' positions only
    CP  predators see no other predators

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

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import train
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.config.config_ppo import RUNS_DIR

ITERATIONS = 2000

CONDITIONS = {
    "A": {},
    "B": {"prey_observe_neighbor_movement": False},
    "C": {"prey_observe_neighbors": False},
    "BP": {"predator_observe_neighbor_movement": False},
    "CP": {"predator_observe_neighbors": False},
}

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in CONDITIONS:
        raise SystemExit(f"usage: launch.py <{'|'.join(CONDITIONS)}> <seed>")
    condition, seed = sys.argv[1], int(sys.argv[2])
    train.config_env.update(CONDITIONS[condition])
    run = f"{{run_name}}_{condition}_seed{seed}"
    train.config_ppo.update(
        iterations=ITERATIONS,
        init_checkpoint=None,
        frozen_policies=[],
        num_env_runners=14,
        seed=seed,
        checkpoint_dir=str(RUNS_DIR / run / "checkpoint"),
        tensorboard_dir=str(RUNS_DIR / run / "tensorboard"),
    )
    sys.argv = sys.argv[:1]  # train.py takes no command-line arguments
    train.main()
