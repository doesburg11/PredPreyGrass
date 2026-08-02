config_env = {
    "seed": 41,
    "max_steps": 1000,
    # Grid and Observation Settings
    "grid_size": 25,
    "num_obs_channels": 3,
    "predator_obs_range": 7,
    "prey_obs_range": 9,
    # Action space settings: 3x3 Moore neighbourhood (8 directions + stay).
    "action_range": 3,
    # Rewards. Only predator_female ever reproduces (males never check the
    # threshold). Hunting/grazing/step rewards default to 0 via _get_role_specific's
    # getattr fallback (unconfigured) -- sparse-reward baseline, matching
    # base_environment_sparse_rewards: only reproduction gives an RL reward signal.
    "reproduction_reward_predator_female": {
        "predator_female": 10.0,
    },
    "reproduction_reward_prey": {
        "prey": 10.0,
    },
    # Energy settings
    "basal_energy_cost_predator_male": 0.15,
    "basal_energy_cost_predator_female": 0.15,
    "basal_energy_cost_prey": 0.05,
    "movement_energy_cost_per_cell_predator": 0.0,
    "movement_energy_cost_per_cell_prey": 0.0,
    "predator_female_creation_energy_threshold": 12.0,
    "prey_creation_energy_threshold": 8.0,
    "initial_energy_predator_male": 5.0,
    # Bumped 5.0 -> 8.0 (2026-08-01, after the first fixed-genome pilot showed
    # episode length capped at ~37-44 steps across every donation rate --
    # suspiciously close to a female's own pure-starvation survival time,
    # 5.0/0.15 =~ 33 steps). This does NOT weaken the obligate-gate guarantee:
    # that guarantee is about net grazing income (female_grass_energy_gain_cap
    # < basal_energy_cost_predator_female), which is independent of initial
    # energy -- she still trends toward zero without gifts, just more slowly,
    # giving gifts more real time to arrive before she starves. Validated:
    # a rate=1.0 pilot with this value reached 34.5 total reproduction events
    # over 60 iterations vs. 0 at rate=0.0. See README.md.
    "initial_energy_predator_female": 8.0,
    "initial_energy_prey": 3.0,
    # Hard cap on energy a predator_female can gain from a single graze event.
    # Set below basal_energy_cost_predator_female (0.15) so that even in the
    # impossible best case of grazing successfully on every single step, a
    # female's net energy income from grazing alone cannot exceed zero -- a
    # mathematical guarantee, not just a statistical tendency, that she cannot
    # reach predator_female_creation_energy_threshold without nuptial gifts.
    # See README.md.
    "female_grass_energy_gain_cap": 0.10,
    # Nuptial-gift genome. Each step a predator_male that successfully hunts
    # donates male_donation_rate * (that step's hunting gain) to predator_female
    # neighbors within cooperation_range. Donation only happens on steps with a
    # successful catch and only when an eligible female neighbor is present
    # (meal-sharing, not a continuous tax on stock energy). Directional: only
    # males donate, only females receive. See README.md.
    "genome_enabled": True,
    # Founder mean bumped 0.0 -> 0.5 (2026-08-02). A founder mean of 0.0 (std
    # 0.05) put the entire starting population deep in the "rate=0" regime the
    # fixed-genome sweep already confirmed produces zero reproduction -- a
    # structural deadlock, not a slow start: mutation only fires at a
    # reproduction event, so a population that never reproduces once can never
    # explore toward a higher, viable rate either. Confirmed directly: seed 42
    # showed zero female_reproduction_events_total through 530/1000 iterations,
    # genome completely static. 0.5 sits between the confirmed-null (0.0) and
    # confirmed-strong (1.0) sweep endpoints, giving the population a realistic
    # chance at early reproduction (so mutation/selection has something to act
    # on at all) while leaving room to observe drift in either direction as a
    # real, control-comparable signal. This does sacrifice the "founder mean=0
    # is a clean neutral baseline" framing used elsewhere in this project, but
    # that framing is moot for an experiment that can't leave the starting
    # gate -- the real-vs-control comparison logic is unaffected by where the
    # shared founder mean sits. Std widened 0.05 -> 0.1 for more immediate
    # heritable variation. See README.md.
    "founder_genome": {
        "predator_male": {
            "male_donation_rate_mean": 0.5,
            "male_donation_rate_std": 0.1,
        },
        # Daughters inherit and can pass on the trait (sex-limited expression --
        # see utils/genome.py), so predator_female needs a founder distribution
        # too even though the trait is inert for females themselves. Identical
        # to predator_male's so the initial population is genetically uniform
        # across sexes.
        "predator_female": {
            "male_donation_rate_mean": 0.5,
            "male_donation_rate_std": 0.1,
        },
        # Prey never reproduce this trait's effect and never mutate meaningfully
        # (mean=0, std=0 -- no variation), but still need a valid founder entry
        # since _policy_group("prey") is a valid genome policy group.
        "prey": {
            "male_donation_rate_mean": 0.0,
            "male_donation_rate_std": 0.0,
        },
    },
    "genome_mutation": {
        "rate": 0.05,
        "std": 0.04,
    },
    "trait_bounds": {
        "male_donation_rate": (0.0, 1.0),
    },
    # Chebyshev-distance radius within which predator_female neighbors are
    # eligible to receive a nuptial gift from a predator_male. Widened 2 -> 4
    # (2026-08-01, same pilot as the initial-energy bump above): under an
    # early, high-entropy, largely-random movement policy, "a male's catch
    # happens to be near a female" is a rare joint event at range 2; widening
    # the eligible radius raises that probability independent of dispersal
    # skill, without changing the underlying obligate-gate energy guarantee
    # at all. See README.md.
    "cooperation_range": 4,
    # Absolute energy caps
    "max_energy_grass": 2.0,
    # Learning agents
    "n_possible_predator_male": 300,
    "n_possible_predator_female": 300,
    "n_possible_prey": 1000,
    "n_initial_active_predator_male": 3,
    "n_initial_active_predator_female": 3,
    "n_initial_active_prey": 8,
    # Grass settings
    "initial_num_grass": 100,
    "initial_energy_grass": 2.0,
    "energy_gain_per_step_grass": 0.04,
    "debug_mode": False,
}
