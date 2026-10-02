import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.calibrate_sarsa import (
    _trajectory_mean,
    run_treatment,
    summarize,
    treatment_specs,
)


def test_treatment_grid_contains_controls_and_sarsa_zero():
    specs = list(treatment_specs(alphas=(0.01,), lambdas=(0.0, 0.9)))
    assert [spec["name"] for spec in specs[:2]] == ["reinforce", "evolution_only"]
    assert any(spec.get("lambda") == 0.0 for spec in specs)
    assert any(spec.get("lambda") == 0.9 for spec in specs)


def test_short_treatment_run_returns_finite_core_metrics(monkeypatch):
    # Shrink the inherited full-world config through the module globals so the
    # test exercises the real runner without paying population-study cost.
    from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa import calibrate_sarsa

    overrides = {
        "grid_size": 12,
        "n_initial_agents": 2,
        "n_initial_carnivores": 0,
        "min_plants": 0,
        "min_trees": 0,
        "wall_interior_density": 0.0,
        "plant_growth_prob": 0.0,
        "tree_birth_prob": 0.0,
        "tree_death_prob": 0.0,
        "carnivore_spawn_interval": 10**9,
        "reproduction_energy_threshold_agent": 10**9,
    }
    monkeypatch.setattr(calibrate_sarsa, "config_sarsa", {**calibrate_sarsa.config_sarsa, **overrides})
    spec = next(
        spec for spec in treatment_specs(alphas=(0.01,), lambdas=(0.9,))
        if spec["algorithm"] == "sarsa"
    )
    row = run_treatment(spec, seed=0, steps=3)
    assert row["completed_steps"] == 3
    assert row["numerical_failure"] == ""
    assert np.isfinite(row["population_mean"])
    assert np.isfinite(row["learning_displacement"])


def test_summary_groups_matched_seeds():
    rows = [
        {
            "treatment": "x", "numerical_failure": "", "extinction_step": "",
            "population_end": 2, "population_mean": 1.5, "births": 1,
            "deaths": 1, "population_peak": 3, "carnivore_peak": 1,
            "learning_displacement": 0.2,
        },
        {
            "treatment": "x", "numerical_failure": "", "extinction_step": 4,
            "population_end": 0, "population_mean": 1.0, "births": 0,
            "deaths": 2, "population_peak": 2, "carnivore_peak": 1,
            "learning_displacement": np.nan,
        },
    ]
    result = summarize(rows)[0]
    assert result["n"] == 2
    assert result["extinctions"] == 1
    assert result["population_end_mean"] == 1.0


def test_extinction_trajectory_is_zero_padded_to_common_horizon():
    # Alive at steps 0 and 1, extinct at step 2, then absorbing zero through 4.
    assert _trajectory_mean([2, 1, 0], requested_steps=4, extinct=True) == 0.6


def test_summary_excludes_partial_numerical_failure_from_outcomes():
    rows = [
        {
            "treatment": "x", "numerical_failure": "", "extinction_step": "",
            "population_end": 10, "population_mean": 8.0, "births": 3,
            "deaths": 1, "population_peak": 12, "carnivore_peak": 2,
            "learning_displacement": 0.2,
        },
        {
            "treatment": "x", "numerical_failure": "overflow", "extinction_step": "",
            "population_end": np.nan, "population_mean": np.nan, "births": 999,
            "deaths": 999, "population_peak": np.nan, "carnivore_peak": np.nan,
            "learning_displacement": np.nan,
        },
    ]
    result = summarize(rows)[0]
    assert result["failures"] == 1
    assert result["population_mean_mean"] == 8.0
    assert result["births_mean"] == 3.0
