"""Tests of evaluation and the fair cooperation measure with untrained
policies on short episodes."""

import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing import evaluate
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.checkpoints import untrained_modules
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import DEFAULT_CONFIG, KINDS, DeerStagHunt
from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.launchers import coop_fair


def short_world():
    config = dict(DEFAULT_CONFIG, max_time_steps=400)
    return DeerStagHunt(config), untrained_modules(config, seed=0)


def test_play_and_summary():
    env, modules = short_world()
    rows = [evaluate.play(env, modules, seed) for seed in (0, 1)]
    for row in rows:
        assert row["physics_steps"] <= 400 and row["stats"]["physics_steps"] == row["physics_steps"]
    again = evaluate.play(env, modules, 0)
    assert again == rows[0]  # seeded: the same episode
    summary = evaluate.summarise(rows)
    assert summary["episodes"] == 2 and sum(summary["outcomes"].values()) == 2
    assert summary["coexisting"] == sum(row["outcome"] == "coexisting" for row in rows)
    assert summary["totals"]["deer_caught"] == sum(row["stats"]["deer_caught"] for row in rows)


def test_coop_fair_counts_and_leaves_the_episode_unchanged():
    env, modules = short_world()
    counts = coop_fair.count(env, modules, 0)
    assert counts["seed"] == 0
    assert counts["hunter_steps"] > 0 and 0 <= counts["with_partner"] <= counts["hunter_steps"]
    for kind in KINDS:
        assert 0 <= counts[f"{kind}_with_partner"] <= counts[f"{kind}_approaches"]
    assert env._physics_step.__func__ is DeerStagHunt._physics_step and env._contacts.__func__ is DeerStagHunt._contacts
    assert evaluate.play(env, modules, 0) == evaluate.play(*short_world(), 0)


def test_coop_fair_rows_carry_their_seed_and_the_summary_ignores_it():
    rows = [dict(seed=s, hunter_steps=10, with_partner=1, fawn_approaches=1, fawn_with_partner=0, middle_approaches=0, middle_with_partner=0, adult_approaches=1, adult_with_partner=1) for s in range(3)]


    assert coop_fair.summarise(rows)["chance"] == 0.1


def test_coop_fair_summary():
    rows = [
        {"hunter_steps": 100, "with_partner": 10, "fawn_approaches": 10, "fawn_with_partner": 0, "middle_approaches": 0, "middle_with_partner": 0, "adult_approaches": 10, "adult_with_partner": 5},
        {"hunter_steps": 100, "with_partner": 30, "fawn_approaches": 0, "fawn_with_partner": 0, "middle_approaches": 5, "middle_with_partner": 1, "adult_approaches": 0, "adult_with_partner": 0},
    ]
    s = coop_fair.summarise(rows)
    assert s["chance"] == 0.2 and s["share"] == {"fawn": 0.0, "middle": 0.2, "adult": 0.5}
    assert s["adult_above_fawn_episodes"] == 1 and s["cooperation_shown"] is True


def test_coop_fair_bootstrap_and_ratio():
    rows = [
        {"hunter_steps": 100, "with_partner": 10, "fawn_approaches": 10, "fawn_with_partner": 0, "middle_approaches": 0, "middle_with_partner": 0, "adult_approaches": 10, "adult_with_partner": 5}
    ] * 10
    s = coop_fair.summarise(rows)
    assert s["adult_over_chance"] == 5.0 and s["adult_minus_chance"] == 0.4
    assert s["adult_minus_chance_interval"] == [0.4, 0.4] and s["adult_above_chance_episodes"] == 10
    empty = [dict(rows[0], adult_approaches=0, adult_with_partner=0)] * 10
    assert coop_fair.summarise(empty)["adult_minus_chance_interval"] is None


def test_verdict_sharing():
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.launchers import verdict_sharing as v

    def run(C, A, coexisting=50):
        return {"coexisting": coexisting, "A": A, "C": C}

    touch = {"touch_1": run(1.12, 14.4), "touch_2": run(1.21, 21.2), "touch_3": run(1.15, 8.0)}
    share = {"share_1": run(1.30, 25.0), "share_2": run(1.22, 22.0), "share_3": run(1.40, 30.0)}
    assert v.judge(share, touch) == {"1_coexistence": True, "2_partnering_C": True, "3_adults_A": True}
    # one sharing run at or below the best touch run fails the ordering
    assert not v.judge(dict(share, share_2=run(1.21, 22.0)), touch)["2_partnering_C"]
    assert not v.judge(dict(share, share_2=run(1.30, 21.0)), touch)["3_adults_A"]
    assert not v.judge(dict(share, share_3=run(1.40, 30.0, 44)), touch)["1_coexistence"]
    assert not v.judge(dict(share, share_1=run(None, 25.0)), touch)["2_partnering_C"]
    with pytest.raises(ValueError):
        v.judge({"share_1": run(1.3, 25)}, touch)


def test_verdict_sharing_loads_only_complete_final_evaluations_of_the_named_run(tmp_path):
    import json

    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.launchers import verdict_sharing as v

    module = tmp_path / "module"
    final = str(module / "runs" / "share_1" / "checkpoint")

    def write(eval_episodes=50, seeds=range(50), coop_episodes=20, coop_seeds=None, checkpoint=final, coop_checkpoint=None, overrides=None):
        rows = [{"seed": s} for s in seeds]
        coop_rows = [{"seed": s} for s in (range(coop_episodes) if coop_seeds is None else coop_seeds)]
        (tmp_path / "share_1_eval.json").write_text(json.dumps({"summary": {"episodes": eval_episodes, "checkpoint": checkpoint, "overrides": overrides or {}}, "episodes": rows}))
        (tmp_path / "share_1_coop.json").write_text(json.dumps({"checkpoint": coop_checkpoint or checkpoint, "overrides": {}, "summary": {"episodes": coop_episodes}, "episodes": coop_rows}))

    write()
    v.load(tmp_path, "share_1", module)
    # deer_stag_hunt's coop rows have no seeds: accepted only when told so
    write(coop_seeds=[None] * 20)
    v.load(tmp_path, "share_1", module, coop_seeded=False)
    with pytest.raises(ValueError):
        v.load(tmp_path, "share_1", module)
    for kwargs in (
        {"eval_episodes": 49, "seeds": range(49)},
        {"seeds": range(1, 51)},
        {"coop_episodes": 19},
        {"coop_seeds": range(1, 21)},
        {"checkpoint": final + "/iter_000500"},
        {"coop_checkpoint": str(module / "runs" / "share_2" / "checkpoint")},
        {"checkpoint": str(module / "runs" / "share_2" / "checkpoint")},
        {"overrides": {"adult_share_fraction": 0.0}},
    ):
        write(**kwargs)
        with pytest.raises(ValueError):
            v.load(tmp_path, "share_1", module)


def test_verdict_sharing_checks_each_world(tmp_path):
    import json

    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.environment import DEFAULT_CONFIG
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.launchers import verdict_sharing as v

    def run(name, **world):
        (tmp_path / name / "source_code").mkdir(parents=True)
        (tmp_path / name / "source_code" / "settings.json").write_text(json.dumps({"config_env": dict(DEFAULT_CONFIG, **world), "config_ppo": {}}))
        (tmp_path / name / "checkpoint").mkdir()
        return {"checkpoint": str(tmp_path / name / "checkpoint")}

    sharing = dict(deer_catch_rule="touch", adult_share_fraction=0.5, adult_share_radius=128.0)
    shared = run("shared", **sharing)
    unshared = run("unshared", deer_catch_rule="touch", adult_share_radius=7.0)  # radius unused without sharing
    party = run("party", adult_share_fraction=0.5, adult_share_radius=128.0)
    hungrier = run("hungrier", **sharing, energy_grass_gain=1.0)
    more_hunters = run("more_hunters", deer_catch_rule="touch", predator_count=8)
    v.check_world(shared, "shared")
    v.check_world(unshared, "unshared", shared=False)
    for summary, shared_world in ((unshared, True), (party, True), (shared, False), (hungrier, True), (more_hunters, False)):
        with pytest.raises(ValueError):
            v.check_world(summary, "x", shared=shared_world)


def test_verdict_sharing_reports_bystanders_per_shared_catch():
    from predpreygrass.non_evolutionary.continuous_space.deer_food_sharing.launchers import verdict_sharing as v

    summary = {"coexisting": 50, "episodes": 50, "totals": {"adults_caught": 100, "adults_shared": 40, "share_bystanders": 60}}
    out = v.values(summary, {"adult_over_chance": 1.3, "chance": 0.1})
    assert out["A"] == 2.0 and out["adults_shared"] == 40 and out["bystanders_per_shared"] == 1.5
    assert v.values(dict(summary, totals={"adults_caught": 0}), {"adult_over_chance": None, "chance": 0.1})["bystanders_per_shared"] is None
