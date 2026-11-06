"""Tests of the heritable rabbit genes (environment.py, rabbit_genes)."""

import json
import math

import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import rules
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import StagHuntRulePrey
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.geometry import Vector

GENES = dict(rabbit_genes=True, mammoth_count=0, steppe_count=0, grass_count=0)


def env_of(**config):
    env = StagHuntRulePrey(dict(GENES, **config))
    env.reset(seed=0)
    return env


def test_without_genes_rabbits_have_the_species_settings():
    env = StagHuntRulePrey(dict(mammoth_count=0, steppe_count=0))
    env.reset(seed=0)
    for rabbit in env.rabbits:
        assert rabbit.genome is None
        assert rabbit.flee_distance == 120.0 and rabbit.birth_threshold == 8.0


def test_the_first_founders_are_the_base_rabbit():
    env = env_of()
    for rabbit in env.rabbits:
        assert rabbit.genome == (0.0, 0.0)
        assert rabbit.flee_distance == pytest.approx(120.0) and rabbit.birth_threshold == pytest.approx(8.0)


def test_phenotypes_follow_the_genes_within_their_limits():
    env = env_of()
    assert env._phenotype((math.log(2), math.log(1.25))) == pytest.approx((240.0, 10.0))
    # Flee at most the view distance; breeding between initial + 0.5 and the maximum.
    assert env._phenotype(env._clipped((5.0, 5.0))) == pytest.approx((267.0, 16.0))
    assert env._phenotype(env._clipped((-5.0, -5.0))) == pytest.approx((15.0, 3.5))
    low, high = env.gene_limits[0]
    assert env._clipped((9.0, 0.0))[0] == high and env._clipped((-9.0, 0.0))[0] == low


def test_a_rabbit_gives_birth_at_its_own_threshold_and_the_child_inherits():
    env = env_of(rabbit_count=1, rabbit_gene_mutation_std=0.0, energy_rabbit_resting_metabolic_cost=0.0, energy_rabbit_speed_cost=0.0)
    env.rabbits[0] = env._animal("rabbit", 0, Vector(400, 400), (0.1, math.log(10 / 8)))
    mother = env.rabbits[0]
    mother.energy = 9.0
    env._reproduce({}, {})
    assert len(env.rabbits) == 1  # 9 < 10: no birth, where the base rabbit would
    mother.energy = 10.0
    env._reproduce({}, {})
    assert len(env.rabbits) == 2
    child = env.rabbits[1]
    assert child.genome == mother.genome
    assert child.birth_threshold == pytest.approx(10.0)
    assert mother.energy == pytest.approx(7.0)


def test_mutation_changes_the_child_within_the_limits():
    env = env_of(rabbit_count=1, rabbit_gene_mutation_std=0.5)
    mother = env.rabbits[0]
    mother.energy = 16.0
    env._reproduce({}, {})
    child = env.rabbits[1]
    assert child.genome != mother.genome
    for g, (low, high) in zip(child.genome, env.gene_limits):
        assert low <= g <= high


def test_neutral_genes_are_inherited_but_not_expressed():
    env = env_of(rabbit_genes_expressed=False)
    rabbit = env._animal("rabbit", 99, Vector(0, 0), (0.5, 0.5))
    assert rabbit.genome == (0.5, 0.5)
    assert rabbit.flee_distance == 120.0 and rabbit.birth_threshold == 8.0


def test_founders_come_from_the_rabbits_alive_at_the_episode_end():
    env = env_of(rabbit_count=4)
    for i, rabbit in enumerate(env.rabbits):
        rabbit.genome = (0.1 * i, -0.1 * i)
    env._end_genes()
    env.reset(seed=1)
    assert sorted(r.genome for r in env.rabbits) == sorted((0.1 * i, -0.1 * i) for i in range(4))
    assert env.rabbits[3].flee_distance == pytest.approx(120 * math.exp(env.rabbits[3].genome[0]))


def test_with_no_rabbits_left_founders_come_from_the_last_to_die():
    env = env_of(rabbit_count=2)
    marked = [(0.2, 0.1), (0.3, 0.1)]
    for rabbit, genome in zip(list(env.rabbits), marked):
        rabbit.genome = genome
        env._remove(rabbit, {}, {}, {}, "starved")
    assert not env.rabbits
    env._end_genes()
    env.reset(seed=1)
    assert sorted(r.genome for r in env.rabbits) == sorted(marked)


def test_a_caught_rabbit_counts_among_the_last_to_die():
    env = env_of(rabbit_count=1)
    rabbit = env.rabbits[0]
    rabbit.genome = (0.4, 0.0)
    hunter = env.predators[0]
    hunter.position = Vector(rabbit.position.x + 5, rabbit.position.y)
    catcher, aways = env._contacts(rabbit)
    env._move(rabbit, 0, catcher, aways, [])
    assert not env.rabbits and list(env.recent_dead) == [(0.4, 0.0)]


def test_fixed_founders_never_change_and_are_not_logged(tmp_path):
    env = env_of(rabbit_count=3, rabbit_gene_founders=[[0.3, 0.2]], rabbit_gene_log=str(tmp_path))
    for rabbit in env.rabbits:
        rabbit.genome = (-0.5, -0.5)
    env._end_genes()
    env.reset(seed=1)
    assert all(r.genome == (0.3, 0.2) for r in env.rabbits)
    assert not list(tmp_path.iterdir())


def test_each_episode_end_is_logged(tmp_path):
    env = env_of(rabbit_count=2, rabbit_gene_log=str(tmp_path))
    for rabbit in env.rabbits:
        rabbit.genome = (0.1, 0.2)
    env._end_genes()
    env.reset(seed=1)
    env._end_genes()
    (log,) = tmp_path.iterdir()
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["episodes"] for line in lines] == [1, 2]
    assert lines[0]["genomes"] == [[0.1, 0.2], [0.1, 0.2]] and lines[0]["expressed"] is True


def test_load_pools_takes_each_lineages_last_pool(tmp_path):
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.launchers.gene_pools import load_pools, use_pool

    (tmp_path / "pool_a.jsonl").write_text(
        json.dumps({"time": 1.0, "genomes": [[0.1, 0.1]]}) + "\n" + json.dumps({"time": 2.0, "genomes": [[0.2, 0.2]]}) + "\n" + json.dumps({"time": 3.0, "genomes": []}) + "\n"
    )
    (tmp_path / "pool_b.jsonl").write_text(json.dumps({"time": 1.0, "genomes": [[0.5, 0.5]]}) + "\n")
    pools = load_pools(tmp_path)
    assert pools == [[[0.2, 0.2]], [[0.5, 0.5]]]
    env = env_of(rabbit_count=2, rabbit_gene_log=str(tmp_path / "unused"))
    use_pool(env, pools, seed=3)
    env.reset(seed=3)
    assert all(r.genome == (0.5, 0.5) for r in env.rabbits)
    assert not (tmp_path / "unused").exists()


def test_the_rule_uses_the_rabbits_own_flee_distance():
    env = env_of(rabbit_count=1, rabbit_fov=360)
    hunter, others = env.predators[0], env.predators[1:]
    hunter.position = Vector(300, 400)
    for other in others:
        other.position = Vector(400, 50)
    base = env._animal("rabbit", 8, Vector(400, 400), (0.0, 0.0))
    short = env._animal("rabbit", 9, Vector(400, 400), (math.log(50 / 120), 0.0))
    assert short.flee_distance == pytest.approx(50.0)
    # The hunter is 100 away: the base rabbit flees, the other wanders (no grass).
    assert rules.prey_move(env, base, env.rule_rng) // rules.DIRECTIONS == rules.FULL
    assert rules.prey_move(env, short, env.rule_rng) // rules.DIRECTIONS == rules.HALF


def test_gene_stats_are_kept_at_the_episode_end():
    env = env_of(rabbit_count=2, max_time_steps=2, action_repeat=1)
    env.rabbits[0].genome = (0.2, 0.0)
    env.step({a: [0, 0] for a in env.active})
    env.step({a: [0, 0] for a in env.active})
    stats = env.episode_gene_stats
    assert stats["rabbit_gene_flee_mean"] == pytest.approx(0.1)
    assert stats["rabbit_gene_pool_size"] == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"rabbit_genes": 1},
        {"rabbit_genes_expressed": "no"},
        {"rabbit_gene_mutation_std": -0.1},
        {"rabbit_gene_log": None},
        {"rabbit_gene_founders": [[0.1]]},
        {"rabbit_gene_founders": []},
        {"rabbit_genes": False, "rabbit_gene_founders": [[0.0, 0.0]]},
    ],
)
def test_gene_settings_are_validated(settings):
    with pytest.raises(ValueError):
        StagHuntRulePrey(dict(GENES, **settings))


def test_gene_arms_start_from_the_lean_fat_weights_and_differ_only_in_expression():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.launchers.calibration import ARMS, PPO_ARMS

    evolving, neutral = dict(ARMS["evolving_rabbits"]), dict(ARMS["neutral_rabbits"])
    assert evolving.pop("rabbit_gene_log") != neutral.pop("rabbit_gene_log")
    assert neutral.pop("rabbit_genes_expressed") is False
    assert evolving == neutral == dict(ARMS["lean_fat_rabbits_more"], rabbit_genes=True)
    assert PPO_ARMS["evolving_rabbits"] == PPO_ARMS["neutral_rabbits"] == PPO_ARMS["lean_fat_rabbits_more"]


def test_an_unfinished_episode_leaves_the_pool_as_it_was():
    env = env_of(rabbit_count=2)
    for rabbit in env.rabbits:
        rabbit.genome = (0.3, 0.3)
    env.reset(seed=1)
    assert all(r.genome == (0.0, 0.0) for r in env.rabbits)


def test_a_finished_episode_updates_the_pool_through_step():
    env = env_of(rabbit_count=2, max_time_steps=2, action_repeat=1)
    for rabbit in env.rabbits:
        rabbit.genome = (0.25, 0.0)
    for _ in range(2):
        env.step({a: [0, 0] for a in env.active})
    env.reset(seed=1)
    assert all(r.genome == (0.25, 0.0) for r in env.rabbits)
    assert env.episode_founders == [(0.25, 0.0), (0.25, 0.0)]


def test_neutral_genes_leave_the_world_exactly_as_without_genes():
    import numpy as np

    def run(**genes):
        env = StagHuntRulePrey(dict(mammoth_count=0, steppe_count=0, max_time_steps=1500, rabbit_gene_mutation_std=0.3, **genes))
        env.reset(seed=7)
        rng = np.random.default_rng(0)
        trace = []
        while True:
            actions = {a: np.array([rng.integers(48), 0]) for a in env.active}
            obs, rewards, terms, truncs, _ = env.step(actions)
            trace.append((sorted(rewards.items()), [(a.id, round(a.position.x, 9), round(a.position.y, 9), round(a.energy, 9)) for a in env.living()]))
            if terms["__all__"] or truncs["__all__"]:
                break
        return trace, env

    plain, _ = run()
    neutral, env = run(rabbit_genes=True, rabbit_genes_expressed=False)
    assert plain == neutral
    assert env.births["rabbit"] > 0  # genes were inherited and mutated along the way
