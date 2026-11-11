"""RESULTS.md section 1's tests: this module's runs share_1-3 (adult catches
shared) against deer_stag_hunt's touch_1-3 (the same world without
sharing), at the final checkpoints.

Reads results/share_<k>_eval.json (evaluate.py, 50 episodes, seeds 0-49)
and results/share_<k>_coop.json (launchers/coop_fair.py, 20 episodes) here,
and deer_stag_hunt's results/touch_<k>_eval.json and touch_<k>_coop.json.

- Test 1 (condition): every sharing run coexists in at least 45 of 50
  episodes.
- Test 2 (primary): every sharing run's adult approaches with a partner near,
  over chance (C), is above every touch run's.
- Test 3 (secondary): the same for adults caught per episode (A).

"All three above all three" is complete separation, the strictest
ordering three runs a side allow. touch_1-3 are historical controls whose
values were known when it was chosen, so it is a pre-registered criterion,
not an exact permutation test (RESULTS.md section 1.3).

Provenance: each evaluation and its coop file must name the same final
checkpoint, of the run named, with no overrides; every setting of each
checkpoint's world is checked: sharing runs must be config/config_env.py's
world, touch runs the same with sharing off (their unused radius aside).
This module's coop_fair.py records each row's seed, which must be 0-19;
deer_stag_hunt's coop rows carry no seed (its coop_fair.py plays seeds 0 to
episodes - 1).

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/deer_food_sharing/launchers/verdict_sharing.py
"""

import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
RESULTS = MODULE / "results"
TOUCH_MODULE = MODULE.parent / "deer_stag_hunt"
TOUCH_RESULTS = TOUCH_MODULE / "results"
REPO = MODULE.parents[3]
SHARE_RUNS = ("share_1", "share_2", "share_3")
TOUCH_RUNS = ("touch_1", "touch_2", "touch_3")
EPISODES = {"eval": 50, "coop": 20}
COEXIST = 45
SHARING = {"deer_catch_rule": "touch", "adult_share_fraction": 0.5, "adult_share_radius": 128.0}


def expected_world(shared):
    """Every setting a checkpoint of each kind must have: this module's world,
    or it with sharing off (the radius then unused, so not checked)."""
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.config.config_env import config_env

    if {k: config_env[k] for k in SHARING} != SHARING:
        raise ValueError("config/config_env.py is not RESULTS.md section 1's world")
    if shared:
        return dict(config_env)
    return {k: v for k, v in dict(config_env, adult_share_fraction=0.0).items() if k != "adult_share_radius"}


def resolve(path):
    """A checkpoint path as saved (relative to the repository root, where
    the launchers run, or absolute), resolved."""
    path = Path(path)
    return (path if path.is_absolute() else REPO / path).resolve()


def load(directory, run, module=None, coop_seeded=True):
    """(evaluation summary, coop summary), checked: complete episodes,
    seeds 0-49, no overrides, and both files naming the same final
    checkpoint, <module>/runs/<run>/checkpoint when module is given."""
    evaluation = json.loads((directory / f"{run}_eval.json").read_text())
    coop = json.loads((directory / f"{run}_coop.json").read_text())
    summary, rows = evaluation["summary"], evaluation["episodes"]
    if summary["episodes"] != EPISODES["eval"] or [row["seed"] for row in rows] != list(range(EPISODES["eval"])):
        raise ValueError(f"{run}: evaluation needs {EPISODES['eval']} episodes with seeds 0-{EPISODES['eval'] - 1}")
    if coop["summary"]["episodes"] != EPISODES["coop"] or len(coop["episodes"]) != EPISODES["coop"]:
        raise ValueError(f"{run}: coop needs {EPISODES['coop']} episodes")
    seeds = [row.get("seed") for row in coop["episodes"]]
    if (coop_seeded or any(s is not None for s in seeds)) and seeds != list(range(EPISODES["coop"])):
        raise ValueError(f"{run}: coop needs seeds 0-{EPISODES['coop'] - 1}")
    if summary.get("overrides") or coop.get("overrides"):
        raise ValueError(f"{run}: evaluated with overrides {summary.get('overrides') or coop.get('overrides')}")
    checkpoint = resolve(summary["checkpoint"])
    if resolve(coop["checkpoint"]) != checkpoint:
        raise ValueError(f"{run}: evaluation and coop name different checkpoints")
    expected = (Path(module) / "runs" / run / "checkpoint").resolve() if module else None
    if checkpoint.name != "checkpoint" or (expected and checkpoint != expected):
        raise ValueError(f"{run}: {checkpoint} is not {expected or 'a final checkpoint'}")
    return summary, coop["summary"]


def values(summary, coop):
    return {
        "coexisting": summary["coexisting"],
        "A": summary["totals"]["adults_caught"] / summary["episodes"],
        "C": coop["adult_over_chance"],
        "S": summary.get("adult_attempt_success"),
        "chance": coop["chance"],
        "births": summary["totals"].get("births_predator", 0) / summary["episodes"],
        "to_bystanders": summary.get("adult_energy_to_bystanders"),
        "adults_shared": summary["totals"].get("adults_shared", 0),
        "bystanders_per_shared": summary["totals"]["share_bystanders"] / summary["totals"]["adults_shared"] if summary["totals"].get("adults_shared") else None,
    }


def above_all(share, touch, key):
    """Every sharing run above every touch run (None counts as not above)."""
    if any(v[key] is None for v in share.values()) or any(v[key] is None for v in touch.values()):
        return False
    return min(v[key] for v in share.values()) > max(v[key] for v in touch.values())


def judge(share, touch):
    if sorted(share) != sorted(SHARE_RUNS) or sorted(touch) != sorted(TOUCH_RUNS):
        raise ValueError("the tests need share_1-3 and touch_1-3")
    return {
        "1_coexistence": all(v["coexisting"] >= COEXIST for v in share.values()),
        "2_partnering_C": above_all(share, touch, "C"),
        "3_adults_A": above_all(share, touch, "A"),
    }


def check_world(summary, run, shared=True):
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.checkpoints import checkpoint_env_config

    config = checkpoint_env_config(resolve(summary["checkpoint"]))
    world = expected_world(shared)
    wrong = {k: (config.get(k), v) for k, v in world.items() if config.get(k) != v}
    if wrong:
        raise ValueError(f"{run}: not this module's world: {wrong}")


def main():
    share, touch = {}, {}
    for run in SHARE_RUNS:
        summary, coop = load(RESULTS, run, MODULE)
        check_world(summary, run)
        share[run] = values(summary, coop)
    for run in TOUCH_RUNS:
        summary, coop = load(TOUCH_RESULTS, run, TOUCH_MODULE, coop_seeded=False)
        check_world(summary, run, shared=False)
        touch[run] = values(summary, coop)
    for name, group in (("sharing", share), ("touch (deer_stag_hunt)", touch)):
        print(name)
        for run, v in group.items():
            print("  " + run + ": " + ", ".join(f"{k} {round(x, 3) if isinstance(x, float) else x}" for k, x in v.items()))
    met = judge(share, touch)
    print(f"tests: {met}")
    if met["1_coexistence"] and met["2_partnering_C"]:
        print("verdict: met (every sharing run partners on adults more than every touch run, and coexists)")
    else:
        print("verdict: not met")
    print(f"secondary, adults caught: {'met' if met['3_adults_A'] else 'not met'}")


if __name__ == "__main__":
    main()
