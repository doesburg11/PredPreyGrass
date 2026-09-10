# Cooperation

This folder groups every `non_evolutionary/` module whose core mechanic is about
**cooperation** between fixed-trait agents: joint/team action, cooperate-vs-defect
dilemmas with free-riding, reputation-conditioned cooperation, reciprocity (direct
and spatial/network), and kin-selection altruism. Every agent trait here is fixed —
only the RL policy adapts, so any cooperative behavior that emerges is a learned
equilibrium under the given incentive design, not a change in population genetics.

Each module keeps its own README with implementation-level detail and (where
available) training results — this file is just the index and the shared framing.

## Modules

* **[`stag_hunt`](stag_hunt)** — the foundational dilemma: cooperative hunting of
  large, energy-rich prey (mammoths) that require pooled predator energy to catch,
  alongside small prey (rabbits) that are usually solo-catchable.
* **[`stag_hunt_defection`](stag_hunt_defection)** — adds an explicit `join_hunt`
  action: cooperate at an energy cost, or defect at zero cost and risk free-riding
  off others' kills.
* **[`stag_hunt_forward_view`](stag_hunt_forward_view)** — `stag_hunt_defection`
  with forward-shifted predator observations; same join/defect mechanic.
* **[`stag_hunt_reputation`](stag_hunt_reputation)** — adds a per-predator
  reputation score built from join/defect history, to test whether cooperation
  becomes conditional on a partner's reputation.
* **[`mammoths`](mammoths)** — the underlying joint/team-capture mechanic on its
  own: prey is only caught if the cumulative energy of surrounding predators
  exceeds it; successful captures split the reward among attackers.
* **[`mammoths_defection`](mammoths_defection)** — adds the same voluntary
  join/free-ride decision as `stag_hunt_defection`, but on top of the `mammoths`
  team-capture mechanic directly.
* **[`shared_prey`](shared_prey)** — the same team-capture-by-pooling mechanic as
  `mammoths`, with the energy ratio flipped (prey typically weaker than a single
  predator rather than stronger).
* **[`direct_reciprocity`](direct_reciprocity)** — every prey is solo-catchable
  (no pooling needed), but predators get a voluntary `share_food` action plus a
  dyadic trust/reciprocity signal, testing whether costly food-sharing emerges
  without any coordination necessity.
* **[`network_reciprocity`](network_reciprocity)** — fixed cooperator/defector
  prey strategies (cooperators donate energy to adjacent prey each step); tests
  whether spatial clustering lets cooperators persist against defectors, per
  Nowak & May (1992).
* **[`lineage_rewards`](lineage_rewards)** — a different cooperation channel:
  kin-selection altruism. Agents are rewarded for their living descendants'
  survival, and fertility-age caps shift older agents from reproducing toward
  protecting existing offspring.

## Fixed-population variant (different ecology, not just a different mechanic)

* **[`pack_hunt_opponent_shaping`](pack_hunt_opponent_shaping)** — same broad
  theme (predator join/free-ride cooperation) as `stag_hunt_defection` and
  `stag_hunt_reputation`, but deliberately **not** built on the full ecology
  those modules share: no reproduction, no death, a fixed predator population
  for the whole run, and scripted (non-learning) prey. That's a second axis of
  "fixed" beyond what the rest of this folder means by it — the population
  itself is fixed, not just each agent's traits — needed to keep the dilemma a
  well-posed repeated game for an N-player LOLA-style opponent-shaping training
  algorithm, which requires a stable set of co-learners to anticipate.
  Environment, a random-policy pygame viewer, the naive-PPO baseline, and a
  from-scratch pairwise N-player opponent-shaping training loop (verified to
  reduce exactly to Foerster2018's own `lola_pg_update` at N=2) all exist and
  are verified; no full training run to convergence has been done yet. See
  its own README for the full reasoning.

## Related but intentionally excluded

* **[`red_queen`](../red_queen)** — the opposite theme: adversarial coevolutionary
  arms-race dynamics between competing prey types, not cooperation.
