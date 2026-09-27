# non_evolutionary

Non-evolutionary experiments: agent traits are fixed at the start of a run and only the RL policy (PPO) adapts. This
folder holds several independent lines of work plus two grouped sub-projects with their own index READMEs
([`project_cooperation/`](project_cooperation/README.md), [`project_reward_shaping/`](project_reward_shaping/README.md)).

## The predator_* lineage: humans, one addition at a time

Three modules build on each other in sequence, each a clone of the previous with earlier results left unchanged.
This is the project's main line for the human-cooperation research angle (predators standing in for humans):

1. **[`predator_sexual_reproduction`](predator_sexual_reproduction/)** — **CLOSED (2026-09-25).** Sexed predators
   (male/female), two-parent reproduction, an asymmetric birth cost, sex-specific hunting odds (males 90% success,
   females 20%), a male-to-mate gift, parental care. Headline result: males learn to hunt and females to gather
   fruit — a learned sexual division of labor, driven mainly by hunting success rate, not death risk. See its
   [`RESULTS.md`](predator_sexual_reproduction/RESULTS.md) for the full 16-iteration writeup and the two questions
   it could not answer.
2. **[`predator_complementary_diet`](predator_complementary_diet/)** — a clone of `predator_sexual_reproduction`
   with one addition: predators need *both* meat and fruit to survive (separate stores, either can starve them),
   `diet_meat_cost_share` tunable. Status as of its README: under calibration, not yet a viable population on its
   own pilots — but it became the base for `predator_bands` regardless.
3. **[`predator_bands`](predator_bands/)** — a clone of `predator_complementary_diet` with **band living** added:
   sharing, kin exclusion, marriage, roaming threats, mammoth hunts, band fission/fusion/drift, and (most recently) a
   free-rider reputation mechanic. This is the active module: see its README's status table and
   [`RESULTS.md`](predator_bands/RESULTS.md) for the running list of what has and hasn't produced learned cohesion.
   Design doc: [`BANDS_DESIGN.md`](predator_complementary_diet/BANDS_DESIGN.md).

## Other modules

| Module | What it is |
|---|---|
| [`base_environment`](base_environment/) | The plain Predator-Prey-Grass environment: no sexes, no diet, no bands — the common ancestor everything else in this folder ultimately traces back to. |
| [`base_environment_seasonal`](base_environment_seasonal/) | `base_environment` + grass regrowth cycling between an "abundant" and a "scarce" phase over an episode. |
| [`base_environment_step_energy`](base_environment_step_energy/) | `base_environment` + a real movement energy cost (the base module charges the same per-step tax regardless of action; this module makes moving itself cost more). See its `RESULTS.md`. |
| [`drive_conditioned_environment`](drive_conditioned_environment/) | `base_environment` + internal drive state conditioning behavior; still close to baseline by design, built incrementally. |
| [`red_queen`](red_queen/) | Tests the Red Queen Hypothesis — an agent must keep adapting to maintain *relative* fitness against a coevolving opponent, not an absolute score. |
| [`walls_occlusion`](walls_occlusion/) | Walls and line-of-sight occlusion experiments. |

## Grouped sub-projects (their own index READMEs)

- **[`project_cooperation/`](project_cooperation/README.md)** — every module whose core mechanic is cooperation
  between fixed-trait agents: joint/team action, cooperate-vs-defect dilemmas with free-riding,
  reputation-conditioned cooperation, direct/spatial/network reciprocity, kin-selection altruism (direct_reciprocity,
  lineage_rewards, mammoths, mammoths_defection, network_reciprocity, pack_hunt_opponent_shaping, shared_prey,
  stag_hunt and its variants).
- **[`project_reward_shaping/`](project_reward_shaping/README.md)** — one connected investigation into sparse vs.
  dense reward shaping on `base_environment`; five trained sibling environments; headline result: reward shaping
  should be minimized here, not maximized.
