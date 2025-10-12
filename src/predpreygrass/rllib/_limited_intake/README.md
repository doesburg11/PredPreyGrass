
# Limited intake


## Multi-step Eating: max_eating_predator and max_eating_prey

---
**Migration note:**

As of October 2025, the environment uses unified intake cap keys:

- `max_eating_predator` (was `max_energy_gain_per_prey`)
- `max_eating_prey` (was `max_energy_gain_per_grass`)

All logic, configs, and docs now use only the new keys. Old keys remain only in legacy experiment folders for reproducibility.
---

**Config keys:**

| Key                  | Default | Description                                                        |
|----------------------|---------|--------------------------------------------------------------------|
| max_eating_predator  | 1.0     | Max energy predator can eat per event (from prey or carcass)        |
| max_eating_prey      | 1.0     | Max energy prey can eat per event (from grass)                      |

Example usage in environment code:

    raw_gain = min(self.agent_energies[caught_prey], self.max_eating_predator)
    raw_gain = min(self.grass_energies[caught_grass], self.max_eating_prey)

If you want the old behavior (unlimited intake per event), set these to a very large value or omit them from the config.



- Adjustments: 
  - Killing prey at a higher energy cost than scavenging for predators?
  - Leave (prey) carcass on grid, as an unmovable energy source for predators.
  - Carcass depletes?
  - Ammount of eating different from killers (more?) than to scavengers (less?)
  - introduce chance of not killing a prey, but nevertheless loose enrgy for attempting?
  - Grass needs probably not much adjusted if it only gets eated partially


## Carcass mechanic (predator over-eat leftovers)

* When a predator catches a prey whose energy exceeds max_eating_predator, the excess energy is left behind as a carcass on the same cell.
- Carcasses are represented in a dedicated observation channel appended after the existing dynamic channels (predators, prey, grass). This increases the base channel count by +1.
- Multiple carcass deposits at the same cell merge their energies.
* Predators can consume carcass energy in subsequent steps: they may take up to max_eating_predator raw energy per step from the carcass, multiplied by energy_transfer_efficiency. The carcass is removed when its energy reaches zero.
- Carcass energy appears in observations subject to the same line-of-sight masking rules as other dynamic channels when mask_observation_with_visibility is enabled.

### Carcass decay and lifetime (new)

- Configurable keys:
  - carcass_decay_per_step: float (default 0.0). Subtracted from each carcass's energy every step before engagements; clipped to available energy.
  - carcass_max_lifetime: int or None (default None). Maximum age in steps; when reached, carcass is removed regardless of remaining energy.
- Internally we track carcass_ages and remove carcasses upon depletion or expiration. Grid channel is updated accordingly.

### Carcass metrics

Per-step and cumulative counters are exposed via env.get_carcass_metrics() and are also attached to infos[agent]["carcass_metrics"] each step:
- created_count, created_energy, consumed_energy, decayed_energy, expired_count, removed_count.
- Also includes active_carcasses and total_carcass_energy snapshot.

See evaluator script: src/predpreygrass/rllib/limited_intake/evaluate_carcass_activity.py to run a quick rollout and plot carcass activity over time.


