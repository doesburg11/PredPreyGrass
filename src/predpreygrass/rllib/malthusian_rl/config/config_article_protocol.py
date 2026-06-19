"""
Protocol configs for article-task reconstructions.

These configs target the original paper tasks, not the Predator-Prey-Grass
analogue. Values marked in `unpublished_reconstruction_defaults` are necessary
to run the environments but are not published in the paper.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

PAPER_URL = "https://www.ifaamas.org/Proceedings/aamas2019/pdfs/p1099.pdf"

RELATED_OFFICIAL_SOURCES = {
    "melting_pot_allelopathic_harvest": {
        "repository": "google-deepmind/meltingpot",
        "checked_refs": ["main", "v1.0.4"],
        "config_paths": [
            "meltingpot/configs/substrates/allelopathic_harvest.py",
            "meltingpot/python/configs/substrates/allelopathic_harvest.py",
        ],
        "component_path": "meltingpot/lua/levels/allelopathic_harvest/components.lua",
        "is_exact_2019_source": False,
        "relation": (
            "Official DeepMind/Lab2D related Allelopathic Harvest substrate, "
            "first described in a 2020 paper, not the two-shrub 2019 "
            "Malthusian RL Allelopathy task."
        ),
        "observed_constants": {
            "num_players": 16,
            "num_berry_types": 3,
            "num_players_upper_bound": 60,
            "episode_timesteps": 2000,
            # Fetched from allelopathic_harvest.py (base closed-world variant):
            "map_width": 32,
            "map_height": 30,
            # Fetched from components.lua:
            "regrowth_minimum_time_to_ripen": 5,   # default in Lua
            "regrowth_base_rate": 0.0000025,        # baseRate default in Lua
            "regrowth_cubic_rate": 0.000009,        # cubicRate default in Lua
            "growth_model": "cubic_positive_autocorrelation",
            "reward_most_tasty": 2,
            "default_other_berry_reward": 1,
            "action_count": 11,
        },
        "why_not_exact": [
            "2019 Allelopathy has two shrub types A/B; Melting Pot has three berry types.",
            "2019 reward grows with repeated same-type harvest up to 250; Melting Pot rewards most-tasty berries with 2 and others with 1.",
            "2019 biased condition has A max reward 8 and B max reward 250; Melting Pot has no matching A/B cap mechanic.",
            "2019 dynamic-population condition uses K=960 and NI=60; Melting Pot default has 16 players in one substrate episode.",
            "2019 paper plots switching-cost counts; Melting Pot tracks berry eating/coloring/zapping statistics instead.",
            "2019 paper growth mechanic: P(grow) inversely proportional to nearby other-type count (cross-type suppression). "
            "Melting Pot uses cubic positive-autocorrelation growth based on same-type count — opposite direction of dependency.",
            "Melting Pot map is 32×30 (closed world) / 31×30 (open world); 2019 paper does not publish map size. "
            "This reconstruction uses 32×32 (square, width from Melting Pot base variant).",
        ],
    },
}

ARTICLE_ALLELOPATHY_BIASED = {
    "task": "allelopathy",
    "variant": "biased",
    "num_species": 4,
    "total_individuals": 960,
    "num_islands": 60,
    "episode_horizon": 1000,
    "alpha": 0.0001,
    "eta": 0.01,
    "resource_spawn_probabilities": [0.8, 0.2],
    "resource_reward_caps": [8, 250],
    "observation_window": 15,
    "deterministic_reset_sequence": True,
    "seed": 0,
    "published_values": {
        "episode_horizon": "Figure 3 caption: episodes lasted 1000 behavior steps.",
        "alpha_eta": "Figure 3 caption: biased Allelopathy alpha=0.0001, eta=0.01.",
        "num_species": "Section 3.2.1: heterogeneous case L=4.",
        "total_individuals": "Section 3.2.1: total K=960.",
        "num_islands": "Section 3.2.1: NI=60.",
        "reward_caps": "Section 3.2: biased A max reward 8, B max reward 250.",
        "observation_window": "Section 2.4: individuals observe a 15x15 RGB window.",
    },
    "unpublished_reconstruction_defaults": {
        # 32×32: square grid. The DeepMind Melting Pot allelopathic_harvest
        # substrate (the closest known related public source) uses 32 columns
        # and 30 rows. The 2019 paper does not publish map dimensions.
        "height": 32,
        "width": 32,
        # 8% density: keeps the open field sparse but with enough shrubs to
        # provide meaningful initial foraging. No published reference.
        "initial_shrub_density": 0.08,
        # 0.01 per empty cell per step: at K/NI=16 agents/island the equilibrium
        # density is roughly 20-30%, providing sustained foraging dynamics.
        # The paper gives the suppression formula (inverse of nearby other-type
        # count) but not the base rate. No published reference.
        "shrub_growth_base_probability": 0.01,
        # Radius 2 → 5×5 neighbourhood (24 cells) around each candidate cell.
        # Provides local suppression without global map-level influence.
        # No published reference.
        "suppression_radius": 2,
        # 4:1 ratio (A:B). Paper says type A is "significantly more common"
        # in the biased variant (Section 3.2). No published exact values.
        "resource_spawn_probabilities": [0.8, 0.2],
    },
    "paper_url": PAPER_URL,
}

ARTICLE_ALLELOPATHY_UNBIASED = {
    **deepcopy(ARTICLE_ALLELOPATHY_BIASED),
    "variant": "unbiased",
    "alpha": 1e-7,
    "eta": 0.3,
    "resource_spawn_probabilities": [0.5, 0.5],
    "resource_reward_caps": [250, 250],
    "published_values": {
        **ARTICLE_ALLELOPATHY_BIASED["published_values"],
        "alpha_eta": "Figure 3 caption: unbiased Allelopathy alpha=1e-07, eta=0.3.",
        "resource_spawn_probabilities": "Section 3.2: unbiased A and B appear with equal probability.",
    },
    "unpublished_reconstruction_defaults": {
        # Inherit map size, density, growth probability, and suppression radius
        # from the biased defaults (all still unpublished for the unbiased case).
        **{
            k: v
            for k, v in ARTICLE_ALLELOPATHY_BIASED["unpublished_reconstruction_defaults"].items()
            if k != "resource_spawn_probabilities"
        },
        # resource_spawn_probabilities for the unbiased variant ([0.5, 0.5]) is
        # PUBLISHED (Section 3.2: "equal probability"), so it does not belong here.
    },
}

ARTICLE_CLAMITY = {
    "task": "clamity",
    # Figure 2(E) legend reads "Malthusian RL (L≥1), dynamic population size."
    # L=1 is the simplest reconstruction default consistent with this. The
    # Clamity section does not test specialisation (unlike Allelopathy L=4);
    # the key comparison is single-agent vs Malthusian multi-agent exploration.
    "num_species": 1,
    "total_individuals": 960,
    "num_islands": 1,
    "episode_horizon": 250,
    "alpha": 0.0001,
    "eta": 1.5,
    "height": 36,
    "width": 60,
    "observation_window": 15,
    "deterministic_reset_sequence": True,
    "seed": 0,
    "published_values": {
        "height_width": "Section 3.1: map size 36x60.",
        "observation_window": "Section 3.1 and Section 2.4: 15x15 window.",
        "episode_horizon": "Figure 2 caption: episodes lasted 250 behavior steps.",
        "alpha_eta": "Figure 2 caption: alpha=0.0001, eta=1.5.",
        "individuals_per_species": "Section 3.1.1: M=960 individuals of each species in the archipelago.",
        "num_species_interpretation": (
            "Figure 2(E): legend reads 'Malthusian RL (L≥1)'; L=1 adopted as "
            "simplest reconstruction default consistent with the notation."
        ),
    },
    "unpublished_reconstruction_defaults": {
        # Radius 4: consistent with Figure 2(B) visual (settled shell spans
        # ~25% of map height → diameter ~9 cells → radius ~4-5) and confirmed
        # by Figure 2(E) reward scale (see base_filter_reward_rate derivation).
        "shell_max_radius": 4,
        # Four symmetric patches estimated from Figure 2(A) screenshot.
        # Map is 36×60, spawn center ~(18, 30). All patches are >10 L1 steps
        # from center (Section 3.1 says ">10 steps away").
        # (6,10): L1=32; (6,49): L1=31; (29,10): L1=31; (29,49): L1=30.
        "nutrient_patches": [(6, 10), (6, 49), (29, 10), (29, 49)],
        # 0.01: derived from Figure 2(E) reward scale. The "no-curiosity"
        # single agent (stuck at local optimum: settle immediately, no patch)
        # reaches ~200. With shell growing 1/step to r=4:
        #   base × (9 + 25 + 49 + 81 + 246×81) = base × 20,090 ≈ 200
        #   → base_filter_reward_rate ≈ 0.01.
        "base_filter_reward_rate": 0.01,
        "spawn_jitter": 2,
    },
    "paper_url": PAPER_URL,
}

ARTICLE_EXPERIMENT_CONDITIONS = {
    "allelopathy_biased_heterogeneous_dynamic": {
        **deepcopy(ARTICLE_ALLELOPATHY_BIASED),
        "experiment_condition": "heterogeneous_dynamic_population",
        "enable_malthusian_update": True,
        "report_results_from": "archipelago_islands",
    },
    "allelopathy_unbiased_heterogeneous_dynamic": {
        **deepcopy(ARTICLE_ALLELOPATHY_UNBIASED),
        "experiment_condition": "heterogeneous_dynamic_population",
        "enable_malthusian_update": True,
        "report_results_from": "archipelago_islands",
    },
    "allelopathy_biased_homogeneous_dynamic": {
        **deepcopy(ARTICLE_ALLELOPATHY_BIASED),
        "experiment_condition": "homogeneous_dynamic_population",
        "num_species": 1,
        "enable_malthusian_update": True,
        "report_results_from": "archipelago_islands",
    },
    "allelopathy_unbiased_homogeneous_dynamic": {
        **deepcopy(ARTICLE_ALLELOPATHY_UNBIASED),
        "experiment_condition": "homogeneous_dynamic_population",
        "num_species": 1,
        "enable_malthusian_update": True,
        "report_results_from": "archipelago_islands",
    },
    "allelopathy_biased_fixed_population_32": {
        **deepcopy(ARTICLE_ALLELOPATHY_BIASED),
        "experiment_condition": "fixed_population_32",
        "num_islands": 30,
        "enable_malthusian_update": False,
        "fixed_population_per_island": 32,
        "report_results_from": "archipelago_islands",
        "published_values": {
            **ARTICLE_ALLELOPATHY_BIASED["published_values"],
            "fixed_population_island_count": (
                "Section 3.2.1: fixed population size 32 required NI=960/32=30."
            ),
        },
    },
    "allelopathy_unbiased_fixed_population_32": {
        **deepcopy(ARTICLE_ALLELOPATHY_UNBIASED),
        "experiment_condition": "fixed_population_32",
        "num_islands": 30,
        "enable_malthusian_update": False,
        "fixed_population_per_island": 32,
        "report_results_from": "archipelago_islands",
        "published_values": {
            **ARTICLE_ALLELOPATHY_UNBIASED["published_values"],
            "fixed_population_island_count": (
                "Section 3.2.1: fixed population size 32 required NI=960/32=30."
            ),
        },
    },
    "clamity_dynamic_population": {
        **deepcopy(ARTICLE_CLAMITY),
        "experiment_condition": "dynamic_population",
        # NI=30 derived: M=960 (published) ÷ 32 agents/island (fixed-population
        # comparison baseline) = 30. Same logic the paper gives explicitly for
        # Allelopathy (Section 3.2.1: "fixed population size 32 required NI=960/32=30").
        "num_islands": 30,
        "num_solitary_eval_islands_per_species": 1,
        "enable_malthusian_update": True,
        "report_results_from": "solitary_eval_islands",
        "published_values": {
            **ARTICLE_CLAMITY["published_values"],
            "solitary_island_protocol": (
                "Section 3.1.1: in parallel with the archipelago, run one solitary island per species; "
                "final results are reported only from solitary islands."
            ),
            "dynamic_population_archipelago": (
                "Section 3.1.1: M=960 individuals of each species appear across the archipelago."
            ),
            "num_islands_derived": (
                "Derived: M=960 (Section 3.1.1) divided by 32 agents/island (fixed-population comparison) "
                "= NI=30. Same derivation the paper gives explicitly for Allelopathy (Section 3.2.1)."
            ),
        },
        "unpublished_reconstruction_defaults": {
            # num_islands=30 is now derived (see published_values above).
            # Remaining unpublished defaults are inherited from ARTICLE_CLAMITY.
            **ARTICLE_CLAMITY["unpublished_reconstruction_defaults"],
        },
    },
    "clamity_fixed_population_32": {
        **deepcopy(ARTICLE_CLAMITY),
        "experiment_condition": "fixed_population_32",
        "total_individuals": 32,
        "num_islands": 1,
        "num_solitary_eval_islands_per_species": 1,
        "enable_malthusian_update": False,
        "fixed_population_per_island": 32,
        "report_results_from": "solitary_eval_islands",
        "published_values": {
            **ARTICLE_CLAMITY["published_values"],
            "fixed_population_32": (
                "Section 3.1.2: fixed population islands of size 32 were compared with dynamic population sizes."
            ),
        },
    },
    "clamity_single_agent_baseline": {
        **deepcopy(ARTICLE_CLAMITY),
        "experiment_condition": "single_agent_baseline_32_replicas",
        "total_individuals": 32,
        "num_islands": 32,
        "num_solitary_eval_islands_per_species": 0,
        "enable_malthusian_update": False,
        "one_agent_per_island": True,
        "report_results_from": "solitary_replicas",
        "published_values": {
            **ARTICLE_CLAMITY["published_values"],
            "single_agent_baseline": (
                "Section 3.1.1: single-agent training sets NI=0 and replicates each solitary island 32 times; "
                "this reconstruction encodes those replicas as 32 one-agent non-Malthusian islands."
            ),
        },
    },
}

ARTICLE_EXACT_BLOCKERS = {
    "claim": "Leibo et al. 2019 exact article reproduction",
    "status": "blocked_without_original_2019_environment_source_or_supplement",
    "source_audit": {
        "paper": PAPER_URL,
        "searched_official_repositories": {
            "google-deepmind/meltingpot": {
                "checked_refs": ["main", "v1.0.4"],
                "result": "related non-identical allelopathic_harvest substrate only",
            },
            "google-deepmind/lab2d": {
                "checked_ref": "0947443",
                "result": "simulator platform source; no Clamity or 2019 Allelopathy task found",
            },
        },
        "official_related_sources": list(RELATED_OFFICIAL_SOURCES),
        "official_related_sources_are_exact": False,
        "no_public_2019_source_found_for": ["Clamity", "2019 two-shrub Allelopathy"],
        "negative_search_terms": [
            "Clamity",
            "clamity",
            "Malthusian",
            "malthusian",
            "trochophore",
            "Trochophore",
        ],
    },
    "missing_published_constants": {
        "allelopathy": [
            "map height and width",
            "initial shrub density/count",
            "base shrub growth probability",
            "suppression radius (nearby defined as within radius r; r value not published)",
            "biased variant exact A/B spawn probabilities",
            "full action semantics beyond the shared movement/rotation description",
            "training horizon in ecological steps for plotted curves",
            "random seeds and sampled entropy/LR values for plotted runs",
        ],
        "clamity": [
            # Estimated from Figure 2(A) screenshot but exact coordinates not published.
            "nutrient patch coordinates/maps from Figure 2",
            # r=4 is consistent with Figure 2(B)+(E) but not explicitly stated.
            "maximum shell size (r=4 confirmed by Figure 2(E) reward scale only)",
            # "food filtering reward rate" resolved: derived from Figure 2(E) → 0.01.
            # "number of archipelago islands" resolved: derived from M=960÷32=30.
            # "number of species L" resolved: interpreted as L=1 from Figure 2(E) "(L≥1)".
            "healthy/unhealthy shell-intersection geometry (L∞ metric is a reconstruction default)",
            "random seeds and sampled entropy/LR values for plotted runs",
        ],
    },
}


def make_article_task_config(
    task: str,
    *,
    variant: str = "biased",
    seed: int = 0,
    condition: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if condition is not None:
        if condition not in ARTICLE_EXPERIMENT_CONDITIONS:
            known = ", ".join(sorted(ARTICLE_EXPERIMENT_CONDITIONS))
            raise ValueError(f"Unknown article experiment condition {condition!r}. Expected one of: {known}.")
        config = deepcopy(ARTICLE_EXPERIMENT_CONDITIONS[condition])
        if config["task"] != task:
            raise ValueError(f"Condition {condition!r} is for task {config['task']!r}, not {task!r}.")
        if task == "allelopathy" and config.get("variant") != variant:
            raise ValueError(f"Condition {condition!r} is for variant {config.get('variant')!r}, not {variant!r}.")
        config["condition_key"] = condition
    elif task == "allelopathy":
        config = deepcopy(
            ARTICLE_ALLELOPATHY_UNBIASED
            if variant == "unbiased"
            else ARTICLE_ALLELOPATHY_BIASED
        )
    elif task == "clamity":
        config = deepcopy(ARTICLE_CLAMITY)
    else:
        raise ValueError("task must be 'allelopathy' or 'clamity'.")

    config["seed"] = int(seed)
    if overrides:
        config.update(overrides)
    return config
