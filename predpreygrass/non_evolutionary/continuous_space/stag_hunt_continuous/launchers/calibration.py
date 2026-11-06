"""Calibration runs (DESIGN.md, plan step 1), trained from scratch with the
settings of config/config_env.py and config/config_ppo.py except:

- `rabbits_only`: no mammoths (step 1a: are hunters viable on rabbits?);
- `rabbits_only_warrens`: rabbits_only with round 12's warrens (can
  rabbits learn to hide?); `rabbits_only_warrens_hidden`: rabbits inside
  hidden from the hunters;
- `with_mammoths`: the configured world (step 1b);
- `with_mammoths_r2` to `with_mammoths_r20`: rounds 2 to 20 of step 1b, see
  ARMS; `rabbits_only_r17` and `rabbits_only_r18` are the first stages of
  rounds 17 and 18, and INIT_FROM names the run an arm continues from;
- `lean_fat_learning`: rabbits_only with lean and fat rabbits, predators
  seeing the rabbits' energy (RESULTS.md, section 5); `lean_fat_big`: the
  same with rabbits breeding at 16 and fat from 12 (section 6).

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
# Round 13 (2026-11-02), after the burrow checks: the discrete stag hunt's
# capture rule (DESIGN.md, 9): a hunt succeeds when the contributors' summed
# energy exceeds ratio x the mammoth's, in round 4 (cap 50, no burrows),
# training truncated at the first prey extinction. Ratio 1 as in the
# discrete module; 1.5 keeps solo kills rarer (6% of round 4's solo
# attempts would succeed at 1.5, 15% at 1).
for arm, ratio in (("with_mammoths_r13a", 1.0), ("with_mammoths_r13b", 1.5)):
    ARMS[arm] = dict(ARMS["with_mammoths_r4"], hunt_rule="energy", hunt_energy_ratio=ratio, truncate_on_prey_extinction=True)
# Round 14 (2026-11-02), after round 13a (ratio 1) kept the mammoths about
# half the hunters' food without a boom, but the rabbits were eaten out
# first in 38 of 50 evaluated episodes: round 13a with round 9's faster
# rabbits (top speed 7, hunters 5).
ARMS["with_mammoths_r14"] = dict(ARMS["with_mammoths_r13a"], rabbit_max_velocity=7.0)
# Round 15 (2026-11-03), after round 14 kept all three species (47 of 50) with
# mammoths ~22% of the hunters' food, and cross-play tied that low share to
# the hunters' policy whichever rabbits they met: round 14 with rabbits a
# poorer food, hunters gaining 0.3 or 0.4 of a caught rabbit's energy (0.5).
for arm, efficiency in (("with_mammoths_r15a", 0.3), ("with_mammoths_r15b", 0.4)):
    ARMS[arm] = dict(ARMS["with_mammoths_r14"], energy_catch_efficiency_predator=efficiency)

# Round 16: round 14 with the rabbits' grass growing in seasons (Grass), so a
# rabbit diet stops being enough for part of each cycle. Settings from playing
# round 14's policies in seasonal worlds.
for arm, period, amplitude in (("with_mammoths_r16a", 2000, 0.5), ("with_mammoths_r16b", 1000, 1.0)):
    ARMS[arm] = dict(ARMS["with_mammoths_r14"], grass_season_period=period, grass_season_amplitude=amplitude)

# Round 17 (2026-11-03), the rabbits-first curriculum (user's idea): hunters
# and rabbits train without mammoths, then both continue with mammoths added,
# whose policy starts untrained (a rabbits-only checkpoint holds its initial
# weights). Round 16's replays kept no more worlds with a delayed first lean
# season or more rabbits at the start, which points at the policies. 17a tests
# the curriculum against round 14, 17b adds round 16b's seasons.
ARMS["rabbits_only_r17"] = dict(ARMS["with_mammoths_r14"], mammoth_count=0)
ARMS["with_mammoths_r17a"] = dict(ARMS["with_mammoths_r14"])
ARMS["with_mammoths_r17b"] = dict(ARMS["with_mammoths_r16b"])
INIT_FROM = {"with_mammoths_r17a": "rabbits_only_r17", "with_mammoths_r17b": "rabbits_only_r17"}

# Round 18 (2026-11-04), after round 17's hunters took a fifth of their food
# from mammoths with or without seasons in the second stage: round 17b with the
# seasons in the first stage too, so hunters learn on rabbits under scarcity.
ARMS["rabbits_only_r18"] = dict(ARMS["with_mammoths_r16b"], mammoth_count=0)
ARMS["with_mammoths_r18"] = dict(ARMS["with_mammoths_r16b"])
INIT_FROM["with_mammoths_r18"] = "rabbits_only_r18"

# Round 19 (2026-11-04), after the energy accounts (payoffs.py) showed ~88% of
# round 14's kills solo, solo hunts losing and pairs paying: round 14 with a
# minimum party of two, at two energy ratios, since a typical pair rarely beats
# a well-fed mammoth at ratio 1. In replay these kept 34-42 of 50. 19c, between
# them, after 19a's hunters left the mammoths (~7% of their food) and 19b boomed
# (3 of 50 at iteration 190, joint arrivals 2.2x chance).
for arm, ratio in (("with_mammoths_r19a", 0.75), ("with_mammoths_r19b", 0.5), ("with_mammoths_r19c", 0.6)):
    ARMS[arm] = dict(ARMS["with_mammoths_r14"], hunt_min_party=2, hunt_energy_ratio=ratio)
# Round 20 (2026-11-04), after 19b's cooperation ate out both prey: 19b with a
# wait of 1000 physics steps between a hunter's births, a brake on hunter
# numbers as human birth spacing is. Rounds 7-8 tried this with solo kills
# possible and the hunters left the mammoths.
ARMS["with_mammoths_r20"] = dict(ARMS["with_mammoths_r19b"], reproduction_predator_interval=1000)
# After the stag hunt line closed (2026-11-06), at the user's request: one
# more run, round 1a (rabbits only, rabbits learning) with the lean and fat
# rabbits of stag_hunt_rule_prey, and predators that see the rabbits' energy
# and their own catch cooldown (RESULTS.md, section 5). From scratch.
ARMS["lean_fat_learning"] = dict(ARMS["rabbits_only"], rabbit_fat_energy=6.0, observe_rabbit_condition=True)
# Its follow-up (RESULTS.md, section 6), from scratch: bigger fat rabbits,
# catches still shared evenly. Rabbits breed at 16 instead of 8 and are fat
# from 12, so a fat rabbit shared by two pays each hunter more than a lean one
# (energy 3 or less) caught alone.
ARMS["lean_fat_big"] = dict(ARMS["lean_fat_learning"], reproduction_rabbit_threshold=16.0, rabbit_fat_energy=12.0)


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
    if arm in INIT_FROM:
        train.config_ppo["init_checkpoint"] = str(RUNS_DIR / f"calibration_{INIT_FROM[arm]}" / "checkpoint")
    if len(sys.argv) > 2:
        train.config_ppo["iterations"] = int(sys.argv[2])
    if len(sys.argv) > 3:
        train.config_ppo["num_env_runners"] = int(sys.argv[3])
    train.main()


if __name__ == "__main__":
    main()
