"""
Frozen paper-protocol mapping for Leibo et al. (2019) reproduction runs.

This is intentionally separate from `config_env.py`. The values here are the
only environment/protocol defaults used by `tune_appo_malthusian_exact.py`.

Important scope boundary: the codebase environment is Predator-Prey-Grass, not
the paper's Clamity or Allelopathy games. This preset freezes the article's
Malthusian protocol mechanics and the closest PPG task analogue, while metadata
below records which article-level pieces remain a mapped approximation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from predpreygrass.rllib.malthusian_rl.config.config_env import config_env

PAPER_SOURCE = {
    "title": "Malthusian Reinforcement Learning",
    "authors": "Leibo et al.",
    "venue": "AAMAS 2019",
    "url": "https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf",
}

PAPER_PROTOCOL_VARIANTS = {
    "allelopathy_unbiased_mapped": {
        "paper_experiment": "Figure 3 A-D, unbiased Allelopathy",
        "alpha": 1e-7,
        "eta": 0.3,
        "smoothing_window_ecological_steps": 25,
        "episode_horizon": 1000,
    },
    "allelopathy_biased_mapped": {
        "paper_experiment": "Figure 3 E-H, biased Allelopathy",
        "alpha": 0.0001,
        "eta": 0.01,
        "smoothing_window_ecological_steps": 25,
        "episode_horizon": 1000,
    },
}

DEFAULT_PAPER_PROTOCOL_VARIANT = "allelopathy_biased_mapped"

paper_protocol_citation_map = {
    "species_policy_sharing": {
        "repo_keys": ["n_possible_type_1_predators", "n_possible_type_2_predators", "n_possible_type_1_prey", "n_possible_type_2_prey"],
        "paper": "Section 2.2 defines a species as individuals sharing one policy network.",
        "status": "mapped",
    },
    "mu_distribution": {
        "repo_keys": ["enable_malthusian_update", "malthusian_mu_update"],
        "paper": "Section 2.3 defines species distributions over islands as softmax-normalized weights.",
        "status": "mapped",
    },
    "ecological_step": {
        "repo_keys": ["max_steps", "enable_within_episode_reproduction"],
        "paper": "Section 2.2 says the ecological scale ticks at the level of single behavioral episodes.",
        "status": "cited",
    },
    "fitness_signal": {
        "repo_keys": ["malthusian_replication_mode", "malthusian_phi_weights"],
        "paper": "Section 2.3 defines individual fitness as exactly cumulative reward and island fitness as the species mean.",
        "status": "cited",
    },
    "population_update_alpha_eta": {
        "repo_keys": ["malthusian_mu_learning_rate", "malthusian_mu_entropy_coeff"],
        "paper": "Figure 3 caption reports alpha/eta for biased and unbiased Allelopathy.",
        "status": "cited",
    },
    "total_population_and_islands": {
        "repo_keys": [
            "n_initial_active_type_1_predator",
            "n_initial_active_type_2_predator",
            "n_initial_active_type_1_prey",
            "n_initial_active_type_2_prey",
            "manual_wall_positions",
        ],
        "paper": "Figure 4 text reports K=960, L=4, M=240, NI=60 for the paper's Allelopathy study.",
        "status": "open_ppg_deviation",
    },
    "task_rewards": {
        "repo_keys": ["reward_predator_catch_prey", "reward_prey_eat_grass", "penalty_prey_caught"],
        "paper": "Section 3.2 describes Allelopathy shrub-harvesting rewards; PPG uses foraging/catching rewards as the closest local analogue.",
        "status": "open_ppg_deviation",
    },
}

acceptance_bands = {
    "mapped_protocol": {
        "claim": "PPG mapped Malthusian protocol reproduced",
        "min_completed_seeds": 3,
        "min_final_training_iteration_fraction": 0.95,
        "max_nan_fraction_core_metrics": 0.05,
        "min_malthusian_metric_columns": 12,
        "min_final_max_island_population": 1.0,
        "required_outputs": [
            "paper_like_metrics.csv",
            "acceptance_report.json",
            "acceptance_report.md",
            "paper_like_collective_return.png",
            "paper_like_population.png",
        ],
    },
    "article_exact": {
        "claim": "Leibo et al. article figures reproduced",
        "status": "blocked_until_unpublished_environment_constants_or_original_source_exist",
        "required_unimplemented_items": [
            "original Clamity constants for Figure 2",
            "original Allelopathy map/growth/spawn constants for Figure 3",
            "Original per-run entropy and learning-rate samples, if exact plotted runs are required",
        ],
    },
}


def make_paper_protocol_env_config(
    *,
    variant: str = DEFAULT_PAPER_PROTOCOL_VARIANT,
    seed: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if variant not in PAPER_PROTOCOL_VARIANTS:
        known = ", ".join(sorted(PAPER_PROTOCOL_VARIANTS))
        raise ValueError(f"Unknown paper protocol variant {variant!r}. Expected one of: {known}.")

    variant_config = PAPER_PROTOCOL_VARIANTS[variant]
    env_config = deepcopy(config_env)
    env_config.update(
        {
            "paper_protocol_name": "leibo_2019_malthusian_protocol_mapped_to_ppg",
            "paper_protocol_variant": variant,
            "paper_protocol_source": PAPER_SOURCE,
            "paper_protocol_scope": "mapped_ppg_protocol_not_original_game",
            "paper_protocol_citation_map": paper_protocol_citation_map,
            "paper_protocol_acceptance_bands": acceptance_bands,
            "seed": 0 if seed is None else int(seed),
            "deterministic_reset_sequence": True,
            "max_steps": int(variant_config["episode_horizon"]),
            "enable_malthusian_update": True,
            "malthusian_replication_mode": "strict",
            "malthusian_mu_update": "multiplicative",
            "malthusian_eta": float(variant_config["eta"]),
            "malthusian_mu_learning_rate": float(variant_config["alpha"]),
            "malthusian_mu_entropy_coeff": float(variant_config["eta"]),
            "malthusian_mu_floor": 0.0,
            "enable_within_episode_reproduction": False,
            "malthusian_phi_weights": {
                "offspring": 0.0,
                "survival": 0.0,
                "foraging": 0.0,
                "energy": 0.0,
                "death": 0.0,
                "reward": 1.0,
            },
            "malthusian_phi_clip": None,
            # PPG reward analogue for article cumulative-reward fitness.
            "reward_predator_catch_prey": {
                "type_1_predator": 1.0,
                "type_2_predator": 1.0,
            },
            "reward_prey_eat_grass": {
                "type_1_prey": 1.0,
                "type_2_prey": 1.0,
            },
            "penalty_prey_caught": {
                "type_1_prey": -1.0,
                "type_2_prey": -1.0,
            },
            "reward_predator_step": {
                "type_1_predator": 0.0,
                "type_2_predator": 0.0,
            },
            "reward_prey_step": {
                "type_1_prey": 0.0,
                "type_2_prey": 0.0,
            },
            "reproduction_reward_predator": {
                "type_1_predator": 0.0,
                "type_2_predator": 0.0,
            },
            "reproduction_reward_prey": {
                "type_1_prey": 0.0,
                "type_2_prey": 0.0,
            },
        }
    )

    if overrides:
        env_config.update(overrides)

    return env_config


config_env_paper_protocol = make_paper_protocol_env_config()
