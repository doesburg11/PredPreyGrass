import math
import json

import pytest

from predpreygrass.non_evolutionary.malthusian_rl.config.config_appo_exact import (
    config_appo_exact,
)
from predpreygrass.non_evolutionary.malthusian_rl.config.config_env import config_env
from predpreygrass.non_evolutionary.malthusian_rl.config.config_article_protocol import (
    ARTICLE_EXPERIMENT_CONDITIONS,
)
from predpreygrass.non_evolutionary.malthusian_rl.config.config_paper_protocol import (
    acceptance_bands,
    config_env_paper_protocol,
    make_paper_protocol_env_config,
)
from predpreygrass.non_evolutionary.malthusian_rl.evaluate_exact_reproduction import (
    derive_paper_like_rows,
    evaluate_article_condition_coverage,
    evaluate_acceptance,
    evaluate_metadata_integrity,
    run_evaluation,
)
from predpreygrass.non_evolutionary.malthusian_rl.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.malthusian_rl.tune_appo_malthusian_exact import (
    build_exact_appo_config,
    validate_exact_configs,
)
from predpreygrass.non_evolutionary.malthusian_rl.utils.reproduction_metadata import (
    build_run_metadata,
)


def _strict_env():
    env = PredPreyGrass(dict(config_env))
    env.reset(seed=123)
    return env


def test_strict_phi_averages_returns_by_species_and_island():
    env = _strict_env()
    island_0, island_1 = sorted(env.island_id_to_cells.keys())[:2]

    env.agents = []
    env.death_agents_stats = {
        "prey-a": {
            "policy_group": "type_1_prey",
            "spawn_island": island_0,
            "cumulative_reward": 4.0,
        },
        "prey-b": {
            "policy_group": "type_1_prey",
            "spawn_island": island_0,
            "cumulative_reward": 2.0,
        },
        "prey-c": {
            "policy_group": "type_1_prey",
            "spawn_island": island_1,
            "cumulative_reward": 10.0,
        },
        "predator-a": {
            "policy_group": "type_2_predator",
            "spawn_island": island_0,
            "cumulative_reward": -1.0,
        },
    }

    phi, counts, components = env._compute_phi_from_episode()

    assert counts["type_1_prey"][island_0] == 2
    assert counts["type_1_prey"][island_1] == 1
    assert counts["type_2_predator"][island_0] == 1
    assert phi["type_1_prey"][island_0] == pytest.approx(3.0)
    assert phi["type_1_prey"][island_1] == pytest.approx(10.0)
    assert phi["type_2_predator"][island_0] == pytest.approx(-1.0)
    assert components["type_1_prey"][island_0]["reward"] == pytest.approx(3.0)


def test_multiplicative_mu_update_matches_hand_computed_softmax():
    env = _strict_env()
    island_ids = sorted(env.island_id_to_cells.keys())
    env.malthusian_mu_update = "multiplicative"
    env.malthusian_mu_learning_rate = 0.5
    env.malthusian_mu_entropy_coeff = 0.0
    env.malthusian_mu_floor = 0.0
    species = "type_1_prey"

    env.mu_logits_by_species[species] = {iid: 0.0 for iid in island_ids}
    env.mu_by_species[species] = {iid: 1.0 / len(island_ids) for iid in island_ids}
    phi_values = {
        island_ids[0]: -1.0,
        island_ids[1]: 1.0,
        island_ids[2]: 0.0,
        island_ids[3]: 0.5,
    }

    env._update_mu_from_phi({species: phi_values})

    logits = [env.malthusian_mu_learning_rate * phi_values[iid] for iid in island_ids]
    denom = sum(math.exp(value) for value in logits)
    expected = {iid: math.exp(logits[idx]) / denom for idx, iid in enumerate(island_ids)}

    for island_id in island_ids:
        assert env.mu_by_species[species][island_id] == pytest.approx(expected[island_id])


def test_exact_appo_config_locks_core_reproduction_invariants():
    validate_exact_configs(config_env_paper_protocol, config_appo_exact)
    appo_config = build_exact_appo_config(config_env_paper_protocol, config_appo_exact)

    assert appo_config.vtrace is True
    assert appo_config.vtrace_clip_rho_threshold == pytest.approx(1.0)
    assert appo_config.vtrace_clip_pg_rho_threshold == pytest.approx(1.0)
    assert appo_config.use_kl_loss is False
    assert appo_config.kl_coeff == pytest.approx(0.0)
    assert appo_config.opt_type == "rmsprop"
    assert appo_config.decay == pytest.approx(0.99)
    assert appo_config.epsilon == pytest.approx(0.0001)
    assert appo_config.num_env_runners == 1
    assert appo_config.num_envs_per_env_runner == 1


def test_exact_config_rejects_drift():
    bad_appo = dict(config_appo_exact)
    bad_appo["vtrace_clip_rho_threshold"] = 2.0

    with pytest.raises(ValueError, match="vtrace_clip_rho_threshold"):
        validate_exact_configs(config_env_paper_protocol, bad_appo)


def test_paper_protocol_env_preset_locks_article_level_protocol_values():
    env_config = make_paper_protocol_env_config(variant="allelopathy_biased_mapped", seed=7)

    assert env_config["seed"] == 7
    assert env_config["deterministic_reset_sequence"] is True
    assert env_config["max_steps"] == 1000
    assert env_config["enable_malthusian_update"] is True
    assert env_config["malthusian_replication_mode"] == "strict"
    assert env_config["malthusian_mu_update"] == "multiplicative"
    assert env_config["malthusian_mu_learning_rate"] == pytest.approx(0.0001)
    assert env_config["malthusian_mu_entropy_coeff"] == pytest.approx(0.01)
    assert env_config["malthusian_phi_weights"]["reward"] == pytest.approx(1.0)
    assert acceptance_bands["mapped_protocol"]["min_completed_seeds"] >= 3


def test_paper_like_evaluator_reconstructs_island_metrics(tmp_path):
    experiment_dir = tmp_path / "APPO_MALTHUSIAN_EXACT_test_seed_0"
    trial_dir = experiment_dir / "trial"
    trial_dir.mkdir(parents=True)
    run_config = {
        "config_env": {"seed": 0, "paper_protocol_variant": "test"},
        "config_appo_exact": {"max_iters": 1},
    }
    (experiment_dir / "run_config.json").write_text(json.dumps(run_config))
    progress_csv = trial_dir / "progress.csv"
    progress_csv.write_text(
        "\n".join(
            [
                "training_iteration,env_runners/episode_return_mean,env_runners/episode_len_mean,"
                "env_runners/malthusian/count/type_1_prey/island_0,"
                "env_runners/malthusian/phi/type_1_prey/island_0,"
                "env_runners/malthusian/count/type_2_prey/island_0,"
                "env_runners/malthusian/phi/type_2_prey/island_0,"
                "env_runners/malthusian/count/type_1_prey/island_1,"
                "env_runners/malthusian/phi/type_1_prey/island_1,"
                "env_runners/malthusian/solitary_return/species_0",
                "1,5.0,1000,2,3.0,1,4.0,5,1.0,42.0",
            ]
        )
        + "\n"
    )

    rows, metadata = derive_paper_like_rows(progress_csv)

    assert metadata["seed"] == 0
    assert metadata["expected_max_iters"] == 1
    assert rows[0]["max_collective_return_over_islands"] == pytest.approx(10.0)
    assert rows[0]["max_per_capita_collective_return_over_islands"] == pytest.approx(10.0 / 3.0)
    assert rows[0]["max_island_population_size"] == pytest.approx(5.0)
    assert rows[0]["solitary_return_mean"] == pytest.approx(42.0)


def test_article_task_acceptance_cannot_be_marked_article_exact():
    rows = []
    metadata = []
    for seed in [0, 1, 2]:
        progress_csv = f"/tmp/article_seed_{seed}/trial/progress.csv"
        rows.append(
            {
                "progress_csv": progress_csv,
                "trial_dir": f"/tmp/article_seed_{seed}/trial",
                "training_iteration": 10,
                "episode_return_mean": 1.0,
                "max_collective_return_over_islands": 2.0,
                "max_per_capita_collective_return_over_islands": 1.0,
                "max_island_population_size": 3.0,
            }
        )
        metadata.append(
            {
                "progress_csv": progress_csv,
                "seed": seed,
                "task": "allelopathy",
                "variant": "biased",
                "expected_max_iters": 10,
                "malthusian_metric_columns": 12,
                "run_config_found": True,
                "metadata_integrity_passed": True,
                "metadata_integrity": {
                    "checksum_present": True,
                    "checksum_valid": True,
                    "environment_snapshot_present": True,
                    "package_versions_present": True,
                    "git_commit_present": True,
                },
            }
        )

    report = evaluate_acceptance(
        rows,
        metadata,
        [
            "paper_like_metrics.csv",
            "acceptance_report.json",
            "acceptance_report.md",
            "paper_like_collective_return.png",
            "paper_like_population.png",
            "paper_like_switching_cost.png",
            "figure3_allelopathy_summary.png",
        ],
    )

    assert report["run_quality_passed"] is True
    assert report["article_exact_passed"] is False
    assert report["article_condition_coverage_passed"] is False
    assert "allelopathy_unbiased_heterogeneous_dynamic" in report["article_condition_coverage"]["missing_conditions"]
    assert report["passed"] is False
    assert report["article_exact"]["status"] == "blocked_without_original_2019_environment_source_or_supplement"


def test_metadata_integrity_requires_valid_checksum_and_environment_snapshot(tmp_path):
    experiment_dir = tmp_path / "APPO_MALTHUSIAN_EXACT_integrity_seed_0"
    trial_dir = experiment_dir / "trial"
    trial_dir.mkdir(parents=True)
    env_config = {"seed": 0, "paper_protocol_variant": "test"}
    appo_config = {"max_iters": 1}
    run_config = {
        "config_env": env_config,
        "config_appo_exact": appo_config,
        **build_run_metadata(env_config, appo_config),
    }
    run_config["config_env"]["seed"] = 1
    (experiment_dir / "run_config.json").write_text(json.dumps(run_config))
    progress_csv = trial_dir / "progress.csv"
    progress_csv.write_text("training_iteration\n1\n")

    _, metadata = derive_paper_like_rows(progress_csv)
    integrity = evaluate_metadata_integrity([metadata])

    assert metadata["metadata_integrity"]["checksum_present"] is True
    assert metadata["metadata_integrity"]["checksum_valid"] is False
    assert integrity["passed"] is False
    assert str(progress_csv) in integrity["failed_progress_csvs"]


def test_article_condition_coverage_requires_all_conditions_and_seed_counts():
    incomplete_metadata = [
        {
            "task": "allelopathy",
            "condition_key": "allelopathy_biased_heterogeneous_dynamic",
            "seed": 0,
        }
    ]

    incomplete = evaluate_article_condition_coverage(
        incomplete_metadata,
        min_seeds_per_condition=3,
    )

    assert incomplete["passed"] is False
    assert "clamity_dynamic_population" in incomplete["missing_conditions"]
    assert "allelopathy_biased_heterogeneous_dynamic" in incomplete["conditions_with_insufficient_seeds"]

    complete_metadata = [
        {
            "task": ARTICLE_EXPERIMENT_CONDITIONS[condition]["task"],
            "condition_key": condition,
            "seed": seed,
        }
        for condition in ARTICLE_EXPERIMENT_CONDITIONS
        for seed in [0, 1, 2]
    ]
    complete = evaluate_article_condition_coverage(
        complete_metadata,
        min_seeds_per_condition=3,
    )

    assert complete["passed"] is True
    assert complete["missing_conditions"] == []
    assert complete["conditions_with_insufficient_seeds"] == []


def _write_synthetic_article_progress(
    root,
    *,
    task: str,
    variant: str,
    condition_key: str,
    experiment_condition: str,
    seed: int,
    solitary_return: float = math.nan,
) -> None:
    experiment_dir = root / f"APPO_MALTHUSIAN_ARTICLE_{task}_{variant}_{experiment_condition}_seed_{seed}"
    trial_dir = experiment_dir / "trial"
    trial_dir.mkdir(parents=True)
    run_config = {
        "config_env": {
            "task": task,
            "variant": variant,
            "condition_key": condition_key,
            "experiment_condition": experiment_condition,
            "report_results_from": "solitary_eval_islands" if task == "clamity" else "archipelago_islands",
            "seed": seed,
        },
        "config_appo_exact": {"max_iters": 1},
    }
    run_config.update(build_run_metadata(run_config["config_env"], run_config["config_appo_exact"]))
    (experiment_dir / "run_config.json").write_text(json.dumps(run_config))
    progress_csv = trial_dir / "progress.csv"
    progress_csv.write_text(
        "\n".join(
            [
                "training_iteration,env_runners/episode_return_mean,env_runners/episode_len_mean,"
                "env_runners/malthusian/count/species_0/island_0,"
                "env_runners/malthusian/phi/species_0/island_0,"
                "env_runners/malthusian/switching_cost/island_0,"
                "env_runners/malthusian/solitary_return/species_0",
                f"1,5.0,1000,2,3.0,4,{solitary_return}",
            ]
        )
        + "\n"
    )


def test_evaluator_writes_condition_and_figure_summaries(tmp_path):
    _write_synthetic_article_progress(
        tmp_path,
        task="clamity",
        variant="biased",
        condition_key="clamity_dynamic_population",
        experiment_condition="dynamic_population",
        seed=0,
        solitary_return=7.0,
    )
    _write_synthetic_article_progress(
        tmp_path,
        task="allelopathy",
        variant="biased",
        condition_key="allelopathy_biased_heterogeneous_dynamic",
        experiment_condition="heterogeneous_dynamic_population",
        seed=0,
    )
    output_dir = tmp_path / "summary"

    report = run_evaluation(
        ray_results_dir=tmp_path,
        experiment_glob="APPO_MALTHUSIAN_ARTICLE_*",
        output_dir=output_dir,
        smooth_window=1,
    )

    assert report["passed"] is False
    assert (output_dir / "condition_summary.csv").exists()
    assert (output_dir / "figure2_clamity_summary.csv").exists()
    assert (output_dir / "figure3_allelopathy_summary.csv").exists()
    assert (output_dir / "figure2_clamity_summary.png").exists()
    assert (output_dir / "figure3_allelopathy_summary.png").exists()
    assert (output_dir / "paper_figure_manifest.json").exists()
    assert "clamity_dynamic_population" in (output_dir / "figure2_clamity_summary.csv").read_text()
    assert "allelopathy_biased_heterogeneous_dynamic" in (output_dir / "figure3_allelopathy_summary.csv").read_text()


# ---------------------------------------------------------------------------
# Policy-mapping drift tests (checklist 3.3)
# ---------------------------------------------------------------------------

def test_ppg_policy_mapping_assigns_species_level_policies():
    """policy_mapping_fn must map every live PredPreyGrass agent to its species policy."""
    from predpreygrass.non_evolutionary.malthusian_rl.tune_appo_malthusian_exact import policy_mapping_fn

    env = PredPreyGrass(dict(config_env))
    env.reset(seed=0)

    expected_policies = {"type_1_predator", "type_2_predator", "type_1_prey", "type_2_prey"}
    seen_policies: set[str] = set()
    for agent_id in env.possible_agents:
        pid = policy_mapping_fn(agent_id)
        assert pid in expected_policies, f"Unexpected policy {pid!r} for agent {agent_id!r}"
        seen_policies.add(pid)

    assert seen_policies == expected_policies, "All four species policies must be reachable"


def test_ppg_policy_mapping_raises_on_malformed_id():
    from predpreygrass.non_evolutionary.malthusian_rl.tune_appo_malthusian_exact import policy_mapping_fn

    with pytest.raises(ValueError, match="Expected agent id"):
        policy_mapping_fn("badid")

    with pytest.raises(ValueError, match="Expected agent id"):
        policy_mapping_fn("only_two")


def test_article_task_policy_mapping_assigns_species_level_policies():
    """article_policy_mapping_fn maps species_X_* → species_X (one policy per species)."""
    from predpreygrass.non_evolutionary.malthusian_rl.tune_appo_article_exact import article_policy_mapping_fn

    assert article_policy_mapping_fn("species_0_agent_3") == "species_0"
    assert article_policy_mapping_fn("species_1_agent_99") == "species_1"
    assert article_policy_mapping_fn("species_3_solitary_0") == "species_3"


def test_article_task_policy_mapping_raises_on_malformed_id():
    from predpreygrass.non_evolutionary.malthusian_rl.tune_appo_article_exact import article_policy_mapping_fn

    with pytest.raises(ValueError, match="Expected agent id"):
        article_policy_mapping_fn("badid")

    with pytest.raises(ValueError, match="Expected agent id"):
        article_policy_mapping_fn("wrong_0_agent_0")  # prefix not "species"


# ---------------------------------------------------------------------------
# Same-seed determinism test (checklist 4.4)
# ---------------------------------------------------------------------------

def test_article_allelopathy_same_seed_produces_identical_episode_summary():
    """ArticleAllelopathyEnv must be fully deterministic across identical seeds."""
    from predpreygrass.non_evolutionary.malthusian_rl.article_tasks import ArticleAllelopathyEnv
    from predpreygrass.non_evolutionary.malthusian_rl.config.config_article_protocol import make_article_task_config

    config = make_article_task_config(
        "allelopathy",
        variant="biased",
        seed=42,
        overrides={
            "num_species": 2,
            "total_individuals": 4,
            "num_islands": 2,
            "episode_horizon": 5,
            "height": 5,
            "width": 5,
            "initial_shrub_density": 0.3,
            "shrub_growth_base_probability": 0.05,
            "deterministic_reset_sequence": True,
        },
    )

    def run_one_episode():
        env = ArticleAllelopathyEnv(config)
        env.reset(seed=42)
        for _ in range(5):
            _, _, _, truncations, infos = env.step({a: 0 for a in env.agents})
            if truncations.get("__all__"):
                return infos["__all__"]
        return {}

    summary_a = run_one_episode()
    summary_b = run_one_episode()

    assert summary_a["switching_cost_by_island"] == summary_b["switching_cost_by_island"]
    for species_key in summary_a["phi_by_species"]:
        for island, phi in summary_a["phi_by_species"][species_key].items():
            assert phi == pytest.approx(summary_b["phi_by_species"][species_key][island])
    for species_key in summary_a["mu_by_species"]:
        for island, prob in summary_a["mu_by_species"][species_key].items():
            assert prob == pytest.approx(summary_b["mu_by_species"][species_key][island])
