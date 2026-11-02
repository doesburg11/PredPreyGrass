"""Calibration runs (DESIGN.md, plan step 1), trained from scratch with the
settings of config/config_env.py and config/config_ppo.py except:

- `rabbits_only`: no mammoths (step 1a: are hunters viable on rabbits?);
- `rabbits_only_warrens`: rabbits_only with round 12's warrens (can
  rabbits learn to hide?); `rabbits_only_warrens_hidden`: rabbits inside
  hidden from the hunters;
- `with_mammoths`: the configured world (step 1b);
- `with_mammoths_r2` to `with_mammoths_r12`: rounds 2 to 12 of step 1b, see ARMS.

Not confirmatory; the seeds of these runs are not reused. Run from the
repository root, e.g. under systemd-run --user:

    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_continuous/launchers/calibration.py \\
        ARM [ITERATIONS] [ENV_RUNNERS]

Output: runs/calibration_<arm>/ (refuses to overwrite).
"""

import sys

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import train
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.config.config_ppo import RUNS_DIR

ARMS = {
    "rabbits_only": {"mammoth_count": 0},
    "with_mammoths": {},
    # Round 2 (2026-10-31), after with_mammoths gave the hunters ~1% of their
    # energy from mammoths: more hunters and mammoths for more pair
    # encounters, and cheaper failed hunts (solo attempt -0.66 instead of
    # -1.65 expected).
    "with_mammoths_r2": {
        "predator_count": 10,
        "mammoth_count": 8,
        "hunt_struggle_fraction": 0.02,
        "hunt_join_cost": 0.02,
    },
    # Round 3 (2026-10-31), after round 2's mammoth share peaked at ~9% of
    # the hunters' energy and fell to ~4%: round 2 with the carcass doubled.
    # The struggle fraction is halved so a failed solo attempt still costs
    # 0.66 (the struggle cost is a fraction of the carcass). A pair's share,
    # 25.6, is above the hunters' energy cap (24), so the cap clips it.
    "with_mammoths_r3": {
        "predator_count": 10,
        "mammoth_count": 8,
        "hunt_carcass_energy": 64.0,
        "hunt_struggle_fraction": 0.01,
        "hunt_join_cost": 0.02,
    },
    # Round 4 (2026-10-31), after round 3 gave the hunters ~90% of their
    # energy from mammoths and both prey were eaten out in ~300 decisions:
    # carcass halfway between rounds 2 and 3, failed solo attempt still 0.66.
    "with_mammoths_r4": {
        "predator_count": 10,
        "mammoth_count": 8,
        "hunt_carcass_energy": 48.0,
        "hunt_struggle_fraction": 0.64 / 48.0,
        "hunt_join_cost": 0.02,
    },
    # Round 5 (2026-10-31), after round 4 also tipped into a hunter boom that
    # ate out both prey (mammoth share ~84%, episodes shortening): round 4
    # with the living-hunter cap halved, so births stop at 25 hunters.
    "with_mammoths_r5": {
        "predator_count": 10,
        "mammoth_count": 8,
        "hunt_carcass_energy": 48.0,
        "hunt_struggle_fraction": 0.64 / 48.0,
        "hunt_join_cost": 0.02,
        "reproduction_max_predator": 25,
    },
    # Round 6 (2026-10-31): limit the hunter boom without a cap, for realism
    # (the user's aim): round 4 (cap 50) with costlier hunter offspring, as
    # in humans: a hunter needs 20 energy to reproduce instead of 12, and a
    # newborn takes 10 from its parent instead of 5 (also the starting
    # energy of the first hunters).
    "with_mammoths_r6": {
        "predator_count": 10,
        "mammoth_count": 8,
        "hunt_carcass_energy": 48.0,
        "hunt_struggle_fraction": 0.64 / 48.0,
        "hunt_join_cost": 0.02,
        "reproduction_predator_threshold": 20.0,
        "energy_predator_initial": 10.0,
    },
    # Round 7 (2026-11-01), after round 6 only delayed the boom (episodes
    # ~450 decisions from iteration ~350; at iteration 920 a hunter's median
    # wait between births was 188 physics steps, 90% under 903): round 4 (cap
    # 50) with a wait of 2000 physics steps between a hunter's births, so at
    # most 4 births per hunter per episode, as human birth spacing limits
    # how fast a population can grow.
    "with_mammoths_r7": {
        "predator_count": 10,
        "mammoth_count": 8,
        "hunt_carcass_energy": 48.0,
        "hunt_struggle_fraction": 0.64 / 48.0,
        "hunt_join_cost": 0.02,
        "reproduction_predator_interval": 2000,
    },
}
# Round 8 (2026-11-01), after round 7 swung the other way (no boom, but
# mammoths ~3% of the hunters' energy and no joint arrivals): shorter waits
# between births, bracketing the window, and training episodes truncated at
# the first prey extinction so that episode length shows coexistence.
for arm, interval in (("with_mammoths_r8a", 1000), ("with_mammoths_r8b", 600)):
    ARMS[arm] = dict(ARMS["with_mammoths_r7"], reproduction_predator_interval=interval, truncate_on_prey_extinction=True)
# Round 9 (2026-11-01), back to round 5 (cap 25), the only world with
# cooperation and the mammoths kept, whose rabbits were nearly all caught
# (152 caught against 145 born per evaluated episode): faster rabbits (top
# speed 6, hunters 5), and training truncated at the first prey extinction.
for arm, speed in (("with_mammoths_r9a", 7.0), ("with_mammoths_r9b", 8.0)):
    ARMS[arm] = dict(ARMS["with_mammoths_r5"], rabbit_max_velocity=speed, truncate_on_prey_extinction=True)
# Round 10 (2026-11-01), after round 9's faster rabbits multiplied into the
# hunters' staple (mammoths ~9% of their food, joint arrivals at chance):
# rabbit burrows (DESIGN.md, 7), 10 of radius 24, in round 5 (cap 25) and
# round 4 (cap 50), training truncated at the first prey extinction.
for arm, base in (("with_mammoths_r10a", "with_mammoths_r5"), ("with_mammoths_r10b", "with_mammoths_r4")):
    ARMS[arm] = dict(ARMS[base], burrow_count=10, burrow_radius=24.0, truncate_on_prey_extinction=True)
# Round 11 (prepared 2026-11-01): well-fed mammoths harder to kill (DESIGN.md,
# 8), in round 4 (cap 50, no burrows, to test the rule alone), training
# truncated at the first prey extinction.
for arm, factor in (("with_mammoths_r11a", 0.5), ("with_mammoths_r11b", 0.8)):
    ARMS[arm] = dict(ARMS["with_mammoths_r4"], hunt_condition_factor=factor, truncate_on_prey_extinction=True)
# Round 12 (2026-11-02), after round 10's rabbits did not use the burrows
# (the median grass patch lay 115 units from the nearest one) and round 11's
# hunters left the mammoths: round 10b (cap 50) with one warren per grass
# cluster, beside it and moving with it (DESIGN.md, 7, point 9).
ARMS["with_mammoths_r12"] = dict(ARMS["with_mammoths_r10b"], burrow_count=20, burrow_placement="cluster")
# Check (2026-11-02), after round 12's rabbits spent ~7% of their time in
# the warrens (~6% untrained) while a hunter boom cut training episodes to
# ~255 decisions: can rabbits learn to hide at all, in the world without
# mammoths where round 1a gave full-length episodes?
ARMS["rabbits_only_warrens"] = dict(ARMS["rabbits_only"], burrow_count=20, burrow_placement="cluster")
# Check 2 (2026-11-02), after check 1's hunters waited at and on the
# warrens (they saw rabbits inside) and no evaluated episode kept both
# species: rabbits inside a warren hidden from the hunters.
ARMS["rabbits_only_warrens_hidden"] = dict(ARMS["rabbits_only_warrens"], burrow_hides_rabbits=True)


def main():
    if not 2 <= len(sys.argv) <= 4 or sys.argv[1] not in ARMS:
        raise SystemExit(f"usage: calibration.py {{{','.join(ARMS)}}} [ITERATIONS] [ENV_RUNNERS]")
    arm = sys.argv[1]
    run = RUNS_DIR / f"calibration_{arm}"
    if run.exists():
        raise SystemExit(f"refusing: {run} exists")
    train.config_env.update(ARMS[arm])
    train.config_ppo["tensorboard_dir"] = str(run / "tensorboard")
    train.config_ppo["checkpoint_dir"] = str(run / "checkpoint")
    if len(sys.argv) > 2:
        train.config_ppo["iterations"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        train.config_ppo["num_env_runners"] = int(sys.argv[3])
    train.main()


if __name__ == "__main__":
    main()
