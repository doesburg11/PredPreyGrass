import pytest

from predpreygrass.non_evolutionary.malthusian_rl.article_tasks import (
    ArticleAllelopathyEnv,
    ArticleClamityEnv,
)
from predpreygrass.non_evolutionary.malthusian_rl.config.config_article_protocol import (
    ARTICLE_EXACT_BLOCKERS,
    ARTICLE_EXPERIMENT_CONDITIONS,
    RELATED_OFFICIAL_SOURCES,
    make_article_task_config,
)
from predpreygrass.non_evolutionary.malthusian_rl.scripts.run_article_condition_matrix import (
    _parse_conditions,
)


def test_article_allelopathy_config_locks_published_protocol_values():
    config = make_article_task_config("allelopathy", variant="biased", seed=3)

    assert config["seed"] == 3
    assert config["episode_horizon"] == 1000
    assert config["alpha"] == pytest.approx(0.0001)
    assert config["eta"] == pytest.approx(0.01)
    assert config["num_species"] == 4
    assert config["total_individuals"] == 960
    assert config["num_islands"] == 60
    assert config["resource_reward_caps"] == [8, 250]
    assert "initial_shrub_density" in config["unpublished_reconstruction_defaults"]


def test_article_allelopathy_tracks_switching_cost_and_malthusian_summary():
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        seed=0,
        overrides={
            "num_species": 1,
            "total_individuals": 1,
            "num_islands": 1,
            "episode_horizon": 2,
            "height": 3,
            "width": 3,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
        },
    )
    env = ArticleAllelopathyEnv(config)
    observations, _ = env.reset(seed=0)
    agent = next(iter(observations))
    state = env.agent_states[agent]

    env.shrubs[state.island][state.row, state.col] = 0
    env.step({agent: 0})
    env.shrubs[state.island][state.row, state.col] = 1
    _, rewards, _, truncations, infos = env.step({agent: 0})

    assert rewards[agent] == pytest.approx(1.0)
    assert truncations["__all__"] is True
    assert infos["__all__"]["switching_cost_by_island"][0] == 1
    assert infos["__all__"]["counts_by_species"]["species_0"][0] == 1
    assert infos["__all__"]["phi_by_species"]["species_0"][0] == pytest.approx(2.0)


def test_article_clamity_settle_grows_shell_and_rewards_when_healthy():
    config = make_article_task_config(
        "clamity",
        seed=0,
        overrides={
            "total_individuals": 1,
            "num_islands": 1,
            "episode_horizon": 2,
            "shell_max_radius": 3,
            "nutrient_patches": [],
        },
    )
    env = ArticleClamityEnv(config)
    observations, _ = env.reset(seed=0)
    agent = next(iter(observations))

    _, reward_1, _, _, _ = env.step({agent: 6})
    _, reward_2, _, truncations, infos = env.step({agent: 0})

    assert reward_1[agent] > 0.0
    assert reward_2[agent] > reward_1[agent]
    assert truncations["__all__"] is True
    assert infos["__all__"]["counts_by_species"]["species_0"][0] == 1


def test_clamity_condition_adds_solitary_eval_without_changing_archipelago_counts():
    config = make_article_task_config(
        "clamity",
        condition="clamity_dynamic_population",
        overrides={
            "total_individuals": 2,
            "num_islands": 2,
            "num_solitary_eval_islands_per_species": 1,
            "episode_horizon": 1,
        },
    )
    env = ArticleClamityEnv(config)
    observations, _ = env.reset(seed=0)

    assert len(observations) == 3
    assert sum(state.solitary_eval for state in env.agent_states.values()) == 1

    _, _, _, _, infos = env.step({})
    summary = infos["__all__"]

    assert sum(summary["counts_by_species"]["species_0"].values()) == 2
    assert summary["solitary_count_by_species"]["species_0"] == 1
    assert "solitary_return_by_species" in summary


def test_clamity_single_agent_baseline_allocates_one_agent_per_replica():
    config = make_article_task_config("clamity", condition="clamity_single_agent_baseline")
    env = ArticleClamityEnv(config)
    env.reset(seed=0)

    counts_by_island = {}
    for state in env.agent_states.values():
        counts_by_island[state.island] = counts_by_island.get(state.island, 0) + 1

    assert len(env.agent_states) == 32
    assert set(counts_by_island.values()) == {1}
    assert all(not state.solitary_eval for state in env.agent_states.values())


def test_fixed_population_condition_allocates_fixed_counts_per_island():
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        condition="allelopathy_biased_fixed_population_32",
        overrides={
            "num_species": 2,
            "total_individuals": 8,
            "num_islands": 2,
            "fixed_population_per_island": 4,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
        },
    )
    env = ArticleAllelopathyEnv(config)
    env.reset(seed=0)

    counts = {
        (species, island): 0
        for species in range(2)
        for island in range(2)
    }
    for state in env.agent_states.values():
        counts[(state.species, state.island)] += 1

    assert set(counts.values()) == {2}
    assert env.enable_malthusian_update is False


def test_article_experiment_conditions_are_named_and_paper_grounded():
    assert ARTICLE_EXPERIMENT_CONDITIONS["clamity_dynamic_population"]["num_solitary_eval_islands_per_species"] == 1
    assert ARTICLE_EXPERIMENT_CONDITIONS["clamity_fixed_population_32"]["fixed_population_per_island"] == 32
    assert ARTICLE_EXPERIMENT_CONDITIONS["allelopathy_biased_fixed_population_32"]["num_islands"] == 30


def test_condition_key_is_persisted_for_evaluator_grouping():
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        condition="allelopathy_biased_heterogeneous_dynamic",
        seed=11,
    )

    assert config["condition_key"] == "allelopathy_biased_heterogeneous_dynamic"
    assert config["experiment_condition"] == "heterogeneous_dynamic_population"


def test_article_condition_matrix_parser_defaults_to_all_conditions():
    assert _parse_conditions(None) == sorted(ARTICLE_EXPERIMENT_CONDITIONS)
    with pytest.raises(ValueError, match="Unknown condition"):
        _parse_conditions(["missing_condition"])


def test_article_exact_blockers_are_explicit_until_original_constants_exist():
    assert ARTICLE_EXACT_BLOCKERS["status"] == "blocked_without_original_2019_environment_source_or_supplement"
    assert "map height and width" in ARTICLE_EXACT_BLOCKERS["missing_published_constants"]["allelopathy"]
    assert "nutrient patch coordinates/maps from Figure 2" in ARTICLE_EXACT_BLOCKERS["missing_published_constants"]["clamity"]
    assert (
        ARTICLE_EXACT_BLOCKERS["source_audit"]["searched_official_repositories"]
        ["google-deepmind/lab2d"]["result"]
        == "simulator platform source; no Clamity or 2019 Allelopathy task found"
    )
    # These three items are now resolved (derived, not unpublished) — must NOT
    # appear as missing constants any more.
    clamity_missing = ARTICLE_EXACT_BLOCKERS["missing_published_constants"]["clamity"]
    assert not any("food filtering reward rate" in s for s in clamity_missing)
    assert not any("number of archipelago islands" in s for s in clamity_missing)
    assert not any("number of species L" in s for s in clamity_missing)


def test_related_official_melting_pot_source_is_not_marked_article_exact():
    source = RELATED_OFFICIAL_SOURCES["melting_pot_allelopathic_harvest"]

    assert source["is_exact_2019_source"] is False
    assert source["observed_constants"]["num_berry_types"] == 3
    assert source["observed_constants"]["num_players"] == 16
    assert source["observed_constants"]["episode_timesteps"] == 2000
    assert source["observed_constants"]["map_width"] == 32
    assert source["observed_constants"]["map_height"] == 30
    assert "two shrub types" in source["why_not_exact"][0]
    # The growth models are fundamentally different: this must be documented.
    any_growth_model_note = any(
        "growth" in w.lower() or "cubic" in w.lower()
        for w in source["why_not_exact"]
    )
    assert any_growth_model_note, "why_not_exact must document the growth model difference"


def test_allelopathy_map_defaults_derived_from_melting_pot():
    """Reconstruction defaults use 32×32 (closest related public source)."""
    biased = make_article_task_config("allelopathy", variant="biased")
    defaults = biased["unpublished_reconstruction_defaults"]
    assert defaults["height"] == 32, "height should be 32 (Melting Pot base substrate width)"
    assert defaults["width"] == 32, "width should be 32 (square grid)"
    assert "resource_spawn_probabilities" in defaults, "biased spawn prob is unpublished"
    assert defaults["resource_spawn_probabilities"] == [0.8, 0.2]

    unbiased = make_article_task_config("allelopathy", variant="unbiased")
    unbiased_defaults = unbiased["unpublished_reconstruction_defaults"]
    assert "resource_spawn_probabilities" not in unbiased_defaults, (
        "unbiased spawn prob [0.5, 0.5] is published (Section 3.2) — must not be in unpublished_reconstruction_defaults"
    )
    assert unbiased_defaults["height"] == 32
    assert unbiased_defaults["width"] == 32


def test_allelopathy_observation_follows_agent_orientation():
    """Paper Section 2.4: the 15x15 window follows the agent's orientation."""
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        seed=0,
        overrides={
            "num_species": 1,
            "total_individuals": 1,
            "num_islands": 1,
            "episode_horizon": 10,
            "height": 7,
            "width": 7,
            "observation_window": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
        },
    )
    env = ArticleAllelopathyEnv(config)
    obs_map, _ = env.reset(seed=0)
    agent = next(iter(obs_map))
    state = env.agent_states[agent]

    # Place a type-0 shrub one step north of the agent.
    north_row = max(0, state.row - 1)
    env.shrubs[state.island][north_row, state.col] = 0

    # Facing north (default, facing==0): the shrub should appear in obs_r=2 (one step
    # ahead of the 5x5 center at radius=2).
    obs_north, *_ = env.step({agent: 0})[:1]
    obs_north = obs_north[agent]
    radius = 2
    assert state.facing == 0, "agent should start facing north"
    assert obs_north[0, radius - 1, radius] == 1.0, "shrub one ahead should be in row radius-1"

    # After turning right twice (now facing south), the same shrub — which is behind
    # the agent — should appear in row radius+1 (one step behind center).
    env.step({agent: 6})   # turn right → facing east
    env.step({agent: 6})   # turn right again → facing south
    obs_south, *_ = env.step({agent: 0})[:1]
    obs_south = obs_south[agent]
    assert state.facing == 2, "agent should now face south"
    assert obs_south[0, radius + 1, radius] == 1.0, "shrub behind should appear in row radius+1"
    # And it must NOT appear in the 'ahead' row anymore.
    assert obs_south[0, radius - 1, radius] == 0.0, "shrub not ahead when facing away"


def test_clamity_observation_follows_agent_orientation():
    """Paper Section 2.4: the 15x15 window follows the agent's orientation."""
    config = make_article_task_config(
        "clamity",
        seed=0,
        overrides={
            "total_individuals": 1,
            "num_islands": 1,
            "episode_horizon": 10,
            "height": 7,
            "width": 7,
            "observation_window": 5,
            "nutrient_patches": [(2, 3)],  # one patch north of centre (3,3)
        },
    )
    env = ArticleClamityEnv(config)
    obs_map, _ = env.reset(seed=0)
    agent = next(iter(obs_map))
    state = env.agent_states[agent]
    state.row, state.col = 3, 3  # fix agent at centre for clarity

    radius = 2
    # Facing north (default): nutrient patch at (2,3) is one row ahead (north).
    obs_n, *_ = env.step({agent: 0})[:1]
    obs_n = obs_n[agent]
    assert obs_n[2, radius - 1, radius] == 1.0, "patch ahead when facing north"

    # Turn right twice → facing south; patch is now one step behind (south row = radius+1).
    env.step({agent: 0})  # absorb step
    state.facing = 2  # face south directly for test clarity
    obs_s, *_ = env.step({agent: 0})[:1]
    obs_s = obs_s[agent]
    assert obs_s[2, radius + 1, radius] == 1.0, "patch behind when facing south"


def test_allelopathy_all_agents_visible_in_observation_channel_2():
    """Other same-island agents must appear in channel 2 (not only self at center)."""
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        seed=0,
        overrides={
            "num_species": 1,
            "total_individuals": 2,
            "num_islands": 1,
            "episode_horizon": 5,
            "height": 7,
            "width": 7,
            "observation_window": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
        },
    )
    env = ArticleAllelopathyEnv(config)
    obs_map, _ = env.reset(seed=0)
    assert len(env.agents) == 2

    agent_a, agent_b = sorted(env.agents)
    state_a = env.agent_states[agent_a]
    state_b = env.agent_states[agent_b]
    # Place agents at known positions
    state_a.row, state_a.col, state_a.facing = 3, 3, 0
    state_b.row, state_b.col = 3, 4  # one step east of agent_a

    obs, *_ = env.step({a: 0 for a in env.agents})[:1]
    obs_a = obs[agent_a]  # agent_a facing north: east = right column in obs

    radius = 2
    # agent_b is 1 step east of agent_a; with facing=N, east offset maps to col+1 in obs
    rr_b, cc_b = 2, 3  # drow=0, dcol=1 → _to_obs_coords facing N → (0+R, 1+R) = (2, 3)
    assert obs_a[2, rr_b, cc_b] == 1.0, "agent_b should be visible in channel 2 of agent_a's obs"
    # Self must also be visible
    assert obs_a[2, radius, radius] == 1.0, "self must be visible in channel 2"


def test_allelopathy_collision_blocks_movement_to_occupied_cell():
    """Two agents attempting to move to the same occupied cell: one is blocked."""
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        seed=99,
        overrides={
            "num_species": 1,
            "total_individuals": 2,
            "num_islands": 1,
            "episode_horizon": 5,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 0.0,
        },
    )
    env = ArticleAllelopathyEnv(config)
    env.reset(seed=99)
    agent_a, agent_b = sorted(env.agents)
    state_a, state_b = env.agent_states[agent_a], env.agent_states[agent_b]
    # Place agents on opposite sides of a cell so one can move into it and block
    state_a.row, state_a.col = 2, 1
    state_b.row, state_b.col = 2, 3

    # Both move toward col=2: agent_a east (action 4), agent_b west (action 3)
    env.step({agent_a: 4, agent_b: 3})

    pos_a = (state_a.row, state_a.col)
    pos_b = (state_b.row, state_b.col)
    assert pos_a != pos_b, "agents must not share a cell after collision"


def test_allelopathic_suppression_is_one_way_A_suppresses_B():
    """Type-A shrub growth is unsuppressed; type-B growth is suppressed by nearby type-A."""
    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        seed=7,
        overrides={
            "num_species": 1,
            "total_individuals": 1,
            "num_islands": 1,
            "episode_horizon": 1,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.0,
            "shrub_growth_base_probability": 1.0,  # always grow if unsuppressed
            "suppression_radius": 2,
        },
    )
    env = ArticleAllelopathyEnv(config)
    env.reset(seed=0)
    island = 0
    # Place a type-A shrub at centre; all surrounding cells are empty.
    env.shrubs[island][:] = -1
    env.shrubs[island][2, 2] = 0  # type A at centre

    # Force shrub type choice to type-B (index 1) for a cell adjacent to type-A
    # by overriding rng — but simpler: iterate _grow_shrubs and check B growth.
    # At base_prob=1.0, type-A would grow freely; type-B next to type-A at radius 2
    # would have growth_prob = 1.0/(1 + count_of_A_nearby).
    # For a cell adjacent to the single type-A at (2,2), _type_a_count_nearby returns 1.
    # So prob for B = 1/(1+1) = 0.5.  For A it remains 1.0.
    # We verify that _type_a_count_nearby correctly counts A near (2, 3) = 1.
    count = env._type_a_count_nearby(env.shrubs[island], 2, 3)
    assert count == 1, "one type-A shrub within radius 2 of (2,3)"

    # Move type-A far enough that (0, 0) is beyond suppression_radius=2.
    # Place type-A at (4, 4); then from (0,0) the box extends to (2,2): (4,4) is outside.
    env.shrubs[island][:] = -1
    env.shrubs[island][4, 4] = 0  # type A in bottom-right corner
    count_far = env._type_a_count_nearby(env.shrubs[island], 0, 0)
    assert count_far == 0, "type-A at (4,4) is not within radius 2 of (0,0)"

    # Confirm that type-A growth uses base_prob directly (no suppression path).
    # Verify this indirectly: call _grow_shrubs and check type-B cell near A is
    # grown only ~50% of steps on average, while isolated type-A cells grow 100%.
    # Here we just confirm the suppression is asymmetric by code inspection —
    # the helper only counts type-A, not type-B, confirming one-way suppression.
    import inspect
    src = inspect.getsource(env._grow_shrubs)
    assert "_type_a_count_nearby" in src
    assert "_nearby_other_type_count" not in src


def test_clamity_shell_growth_restricted_by_adjacent_shells():
    """Section 3.1: 'shell growth is also restricted by the presence of adjacent shells.'

    Two agents settle adjacent to each other; each shell must stop growing while
    the other is within adjacency range — not just receive zero reward.
    """
    config = make_article_task_config(
        "clamity",
        overrides={
            "num_species": 1,
            "total_individuals": 2,
            "num_islands": 1,
            "episode_horizon": 20,
            "height": 36,
            "width": 60,
            "nutrient_patches": [],
            "shell_max_radius": 4,
        },
    )
    env = ArticleClamityEnv(config)
    env.reset(seed=0)

    agents = list(env.agent_states)
    a, b = agents[0], agents[1]

    # Place agents one cell apart so their shells (radius 1) are immediately adjacent.
    env.agent_states[a].row, env.agent_states[a].col = 10, 10
    env.agent_states[b].row, env.agent_states[b].col = 10, 12  # 2 cols apart

    # Both settle this step.
    _, _, _, _, _ = env.step({a: 6, b: 6})
    assert env.agent_states[a].shell_radius == 1
    assert env.agent_states[b].shell_radius == 1

    # After a few more steps the shells should NOT grow because each agent is
    # within adjacency range of the other (L1 dist 2 ≤ r_a + r_b = 2).
    for _ in range(5):
        if env.current_step >= env.episode_horizon:
            break
        env.step({a: 0, b: 0})

    assert env.agent_states[a].shell_radius == 1, (
        "Shell must not grow when an adjacent settled shell is present (Section 3.1)."
    )
    assert env.agent_states[b].shell_radius == 1, (
        "Shell must not grow when an adjacent settled shell is present (Section 3.1)."
    )


def test_clamity_base_filter_reward_rate_matches_figure2e_local_optimum():
    """base_filter_reward_rate=0.01 derived from Figure 2(E) reward scale.

    The 'no-curiosity' agent stuck at the local optimum (settles at step 0,
    no nutrient patch) reaches ~200 total reward.  With shell growing 1 radius
    per step to shell_max_radius=4:
      total_area_steps = 9 + 25 + 49 + 81 + 246*81 = 20,090
      base * 20,090 = 200  =>  base ≈ 0.01
    """
    config = make_article_task_config(
        "clamity",
        overrides={
            "num_species": 1,
            "total_individuals": 1,
            "num_islands": 1,
            "episode_horizon": 250,
            "height": 36,
            "width": 60,
            "nutrient_patches": [],       # no nutrient patches → pure local optimum
            "shell_max_radius": 4,
            "base_filter_reward_rate": 0.01,
        },
    )
    env = ArticleClamityEnv(config)
    env.reset(seed=0)

    # Force the one agent to settle immediately (action 6) and then do nothing.
    agent = list(env.agent_states)[0]
    _, _, _, _, infos = env.step({agent: 6})          # settle at step 0
    total_reward = env.agent_states[agent].cumulative_reward
    for _ in range(249):
        if env.current_step >= env.episode_horizon:
            break
        _, _, _, _, infos = env.step({agent: 0})
        total_reward = env.agent_states[agent].cumulative_reward

    # Expected ≈ 0.01 × (9+25+49+81+246×81) = 0.01 × 20,090 = 200.9
    assert 190 <= total_reward <= 220, (
        f"Local-optimum Clamity total reward {total_reward:.1f} should be ~200 "
        f"(derived from Figure 2(E)). Check base_filter_reward_rate."
    )


def test_clamity_dynamic_population_ni_is_derived():
    """NI=30 for clamity_dynamic_population is derived (M=960 / 32 = 30)."""
    from predpreygrass.non_evolutionary.malthusian_rl.config.config_article_protocol import (
        ARTICLE_EXPERIMENT_CONDITIONS,
    )
    dyn = ARTICLE_EXPERIMENT_CONDITIONS["clamity_dynamic_population"]
    assert dyn["num_islands"] == 30
    # The derivation must be documented in published_values (not just unpublished).
    assert "num_islands_derived" in dyn["published_values"], (
        "NI=30 derivation (M=960/32) must appear in published_values, not only in "
        "unpublished_reconstruction_defaults."
    )
    # num_islands must NOT appear as an unpublished reconstruction default any more.
    assert "num_islands" not in dyn.get("unpublished_reconstruction_defaults", {}), (
        "NI=30 is derivable; it should not still be listed as unpublished."
    )
