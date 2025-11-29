# Nature vs Nurture: Emergence of Cooperation in PredPreyGrass

This document summarizes how **emergent cooperation** appears in your
PredPreyGrass project, and contrasts two modes that map to the classic
**nature-versus-nurture** question:

- **Nature-driven cooperation** – patterns that look stable and well-formed in
  the *final behaviour* (e.g. predators hunting efficiently together), but are
  largely *baked into the design* via reward shaping, environment structure, or
  selection mechanics.
- **Nurture-driven cooperation** – patterns that **have to fight against local
  incentives**, emerging from a tug-of-war between individual and group
  selection, and are therefore more informative about when and why cooperation
  is actually stable.

In programming terms, "nature" is everything you hard-code – reward signals,
termination rules, observation design – while "nurture" is what the agents or
lineages discover despite, or because of, these constraints. The goal here is
to make explicit what your current setups are really testing, and what changes
move you along the spectrum from **nature-heavy** to **nurture-heavy**
emergence.

---

## 1. Cooperation in the RLlib PredPreyGrass Environment

### 1.1 Where cooperation comes from

In `rllib_termination/predpreygrass_rllib_env.py`, cooperation shows up in
several ways:

- **Predators** benefit from coordinating chase patterns and sharing access to
  prey over time.
- **Prey** benefit from spatial dispersion, leading behaviour, and occasional
  altruistic blocking (one prey sacrificing some safety for kin).
- **Grass** introduces resource dynamics that couple predator and prey
  population stability.

However, this cooperation is not purely "discovered" in a vacuum. It is heavily
influenced by **reward design and termination conditions**:

- Rewards depend on events like successful hunts, survival, and reproduction.
- Recent changes in `rllib_termination` route **kin rewards** only when actual
  reproduction happens and the parent is alive.
- Episode termination protocols encode assumptions about when a run is "over"
  (e.g. all predators dead, prey extinct, etc.), which creates extra pressure
  for certain cooperative patterns.

In this sense, RLlib predators and prey exhibit **nature-driven cooperation**: it
appears from learning, but **the reward landscape already encodes a strong
prior** that makes those patterns natural.

### 1.2 Nature-driven cooperation: pros and limits

**Pros**

- Great for **engineering**: if you want stable hunting packs or sustainable
  predator–prey cycles, shaped rewards and tuned termination rules give you
  control.
- Supports **Red Queen**-style experiments (e.g. freezing one side) where you
  can compare learned policies under controlled reward assumptions.

**Limits for emergence questions**

- Because the incentives are already aligned with cooperative behaviour, it is
  hard to tell *how much* cooperation is due to:
  - innate reward design, versus
  - genuine **evolutionary or learning pressure** to coordinate.
- It is difficult to separate "agents are cooperative" from "the designer made
  cooperation the easiest way to get reward".

---

## 2. Multi-Level Selection (MLS) Public-Goods World

To probe emergence in a more controlled way, you introduced a **multi-level
selection (MLS)** world in `multi_level_selection/`:

- `public_goods_sim.py` defines light-weight genomes (baseline, sensitivity,
  reciprocity) for ~200 agents split into ~10 groups.
- `animate_cooperation.py` runs repeated public-goods games and visualizes:
  - global average contribution (as a proxy for cooperation),
  - trait diversity (baseline std),
  - total payoff per group each generation.
- `MLSResultsLogger` writes CSVs into the same Ray-style results tree used by
  PPO experiments (e.g. `run_config.json`, `mls_generation_metrics.csv`,
  `predator_final.csv`).

### 2.1 What the current MLS run shows

In the initial MLS configuration, your plot looked roughly like this:

- **Avg contribution** (blue) rises quickly towards the maximum and stays high.
- **Baseline std** (orange) drops from a higher initial value to a lower,
  wiggly plateau.
- **Group payoffs** are almost identical and high across all groups.

Interpretation:

- The public-goods payoff + group-level selection strongly favour groups with
  high contributions.
- Mutation provides some variation, but selection rapidly pulls the entire
  population toward "high cooperator" genomes.
- There is little persistent between-group variation once cooperation has
  locked in.

This is again mostly **nature-driven cooperation**:

- The **game itself is pro-cooperation**: contributions are multiplied and
  shared, and the cost of contributing is modest compared to the group benefit.
- **Selection is applied at the group level only**: you copy agents from the
  top-performing groups to form the next generation, with no within-group
  competition.
- Free-riders have only a mild advantage locally, and even that is washed out by
  group-level replication.

So, cooperation *does* emerge from random genomes under selection, but the
setup is **already biased toward cooperative outcomes**.

---

## 3. From Nature to Nurture: When Cooperation is Truly Emergent

To move toward **nurture-style emergence**, you want a **tension** between:

- **Individual incentives to defect** ("selfish gene" pressure), and
- **Group incentives to sustain cooperation** (multi-level selection).

### 3.1 Ingredients of nurture emergence

1. **Within-group advantage for defectors**
   - Non-contributors should strictly outperform contributors *within the same
     group*, given the same group context.
   - Example: contributors pay a larger individual cost, while defectors still
     enjoy part of the multiplied public good.

2. **Within-group selection**
   - Copy agents in proportion to their individual payoff *inside* each group,
     not just by selecting entire groups as units.
   - This lets cheaters spread locally even in otherwise successful groups.

3. **Between-group selection**
   - Groups with more cooperators still get higher total payoff, so they are
     more likely to seed offspring groups.
   - Now you have a genuine **multi-level conflict of selection**: defectors win
     short-term inside groups; cooperative groups win long-term across groups.

4. **Mutation and drift**
   - Mutation keeps introducing both more cooperative and more exploitative
     variants, so the system can explore the edge between collapse and
     sustainable cooperation.

### 3.2 Diagnosing where your MLS world sits

Right now, your MLS world is roughly here on the spectrum:

- **Strong group selection**, **weak within-group competition** ⇒ heavily
  nature-driven cooperation.
- To get to nurture-style emergence, you need:
  - *Sharper penalties* for cooperating individuals,
  - *Stronger within-group reproduction* based on individual payoffs, and
  - *Explicit metrics* for the prevalence of defectors.

With those changes, you could see:

- Regimes where cooperation **collapses** under individual-level selection.
- Regimes where multi-level selection **rescues** cooperation by favouring
  cooperative groups, even while cheaters temporarily do well.
- Red Queen regimes where cooperation and defection chase each other in cycles.

---

## 4. How This Connects Back to PredPreyGrass RLlib (the programming dilemma)

The MLS world is intentionally simplified compared to the full RLlib predator–
prey–grass environment, but the conceptual mapping is useful, especially for
highlighting **programmer dilemmas**:

- **Genome traits ↔ policy parameters**
  - Baseline, sensitivity, reciprocity in MLS correspond loosely to how
    predators or prey in RLlib value chasing, sharing, or sacrificing.

- **Within-group vs between-group selection**
  - In RLlib, **gradient updates** are more like within-group selection:
    policies that get higher reward get reinforced.
  - Multi-seed experiments or population-based training (PBT) can act more like
    **between-group selection**, where entire policies are promoted or replaced.

-- **Nature vs nurture framing in code**
  - Reward shaping in `predpreygrass_rllib_env.py` (including kin kickbacks)
    moves you toward **nature-driven** cooperation: you as the programmer bake
    in the idea that certain cooperative behaviours should be rewarded. The
    risk is that you can "solve" the social dilemma in the reward design long
    before the agents ever have to confront it.
  - MLS-style experiments, especially if you strengthen within-group incentives
    to defect, pull you toward **nurture-driven** cooperation: you ask whether
    cooperation can survive when the most locally rational move is to cheat,
    and only group-level or long-horizon dynamics can rescue it.

This creates a **programming dilemma**:

- If you over-design the rewards (heavy nature), you get beautiful cooperative
  behaviour that is hard to interpret scientifically: was it discovered, or was
  it simply the only easy way to get reward?
- If you under-design the rewards (heavy nurture), you risk getting no
  interesting behaviour at all: agents learn degenerate strategies or collapse
  ecosystems.

The dual view is powerful:

- Use **nature-heavy setups** to engineer robust cooperative behaviours (for
  demos, benchmarks, and controlled comparative tests where you value stability
  and reproducibility).
- Use **nurture-heavy setups** to study the conditions under which cooperation
  is actually stable against exploitation, even when local incentives point the
  other way.

---

## 5. Possible Next Steps

Here are concrete directions if you want to push the nurture side further.

### 5.1 Strengthen defector advantage in MLS

- Increase the cost of contributing or make the multiplier depend on the
  *fraction* of contributors so mixed groups are especially vulnerable.
- Log and plot:
  - fraction of agents with low average contribution (defectors),
  - fraction of cooperative vs defective groups over generations.

### 5.2 Add within-group selection to `public_goods_sim.py`

- After each match, reproduce agents *within each group* proportionally to
  their payoffs, then apply group-level selection as a second-stage filter.
- Compare:
  - pure group selection (current setup),
  - pure within-group selection,
  - mixed multi-level selection.

### 5.3 Mirror these ideas in RLlib

- Use **population-based training** or multi-policy setups as the analogue of
  group selection.
- Experiment with **more adversarial reward structures** where individual
  self-interest and group performance pull in different directions, then look at
  whether kin rewards and reproduction still support cooperation.

---

## 5.4 Open Brainstorm: Questions to Explore in Code

Some deliberately open-ended prompts you could explore directly in this
codebase:

- **How fragile is cooperation to tiny reward changes?**
  - Take a working cooperative RLlib setup and *slightly* tweak the reward
    scaling for selfish acts (e.g. lone predator kills, prey hoarding grass).
  - Does cooperation collapse abruptly, fade gradually, or remain robust?
  - This would tell you whether the cooperation you see is resting on a knife
    edge of nature (reward design) or on a broad plateau enforced by nurture
    (learning dynamics).

- **Can you get "fake" cooperation that only lives inside termination rules?**
  - Design a scenario where agents appear cooperative only because early
    termination punishes visible conflict.
  - Then relax the termination condition and see whether the same learned
    policies stay cooperative or immediately exploit the extra time.

- **What happens if you deliberately reward short-term defection but long-term
  group survival?**
  - In RLlib, add a small bonus for highly selfish local actions (nature pulling
    toward cheating) *and* a larger, delayed bonus for group-level metrics
    (e.g. total biomass after many steps).
  - Does PPO discover strategies that balance these, or does it get stuck in
    short-term exploitation?

- **Can MLS runs predict which reward shapings will be dangerous?**
  - Use the public-goods world as a sandbox: treat different payoff tables as
    analogues for different RL reward shapings.
  - Look for regions where multi-level selection consistently fails; avoid
    designing RL rewards that implicitly fall into those regions.

- **Is there a "minimal" reward spec that still yields cooperation?**
  - Gradually strip away shaping terms from `predpreygrass_rllib_env.py`,
  - logging when cooperation first disappears.
  - Each removal is a small programming decision that answers, "Was this
    nature or was this nurture doing the heavy lifting?"

These are not prescriptions, just seeds for experiments. The point is to make
+# Nature vs Nurture: Emergence of Cooperation in PredPreyGrass
+
+This document summarizes how **emergent cooperation** appears in your
+PredPreyGrass project, and contrasts two modes that map to the classic
+**nature-versus-nurture** question:
+
+- **Nature-driven cooperation** – patterns that look stable and well-formed in
+  the *final behaviour* (e.g. predators hunting efficiently together), but are
+  largely *baked into the design* via reward shaping, environment structure, or
+  selection mechanics.
+- **Nurture-driven cooperation** – patterns that **have to fight against local
+  incentives**, emerging from a tug-of-war between individual and group
+  selection, and are therefore more informative about when and why cooperation
+  is actually stable.
+
+In programming terms, "nature" is everything you hard-code – reward signals,
+termination rules, observation design – while "nurture" is what the agents or
+lineages discover despite, or because of, these constraints. The goal here is
+to make explicit what your current setups are really testing, and what changes
+move you along the spectrum from **nature-heavy** to **nurture-heavy**
+emergence.
+
+---
+
+## 1. Cooperation in the RLlib PredPreyGrass Environment
+
+### 1.1 Where cooperation comes from
+
+In `rllib_termination/predpreygrass_rllib_env.py`, cooperation shows up in
+several ways:
+
+- **Predators** benefit from coordinating chase patterns and sharing access to
+  prey over time.
+- **Prey** benefit from spatial dispersion, leading behaviour, and occasional
+  altruistic blocking (one prey sacrificing some safety for kin).
+- **Grass** introduces resource dynamics that couple predator and prey
+  population stability.
+
+However, this cooperation is not purely "discovered" in a vacuum. It is heavily
+influenced by **reward design and termination conditions**:
+
+- Rewards depend on events like successful hunts, survival, and reproduction.
+- Recent changes in `rllib_termination` route **kin rewards** only when actual
+  reproduction happens and the parent is alive.
+- Episode termination protocols encode assumptions about when a run is "over"
+  (e.g. all predators dead, prey extinct, etc.), which creates extra pressure
+  for certain cooperative patterns.
+
+In this sense, RLlib predators and prey exhibit **nature-driven cooperation**: it
+appears from learning, but **the reward landscape already encodes a strong
+prior** that makes those patterns natural.
+
+### 1.2 Nature-driven cooperation: pros and limits
+
+**Pros**
+
+- Great for **engineering**: if you want stable hunting packs or sustainable
+  predator–prey cycles, shaped rewards and tuned termination rules give you
+  control.
+- Supports **Red Queen**-style experiments (e.g. freezing one side) where you
+  can compare learned policies under controlled reward assumptions.
+
+**Limits for emergence questions**
+
+- Because the incentives are already aligned with cooperative behaviour, it is
+  hard to tell *how much* cooperation is due to:
+  - innate reward design, versus
+  - genuine **evolutionary or learning pressure** to coordinate.
+- It is difficult to separate "agents are cooperative" from "the designer made
+  cooperation the easiest way to get reward".
+
+---
+
+## 2. Multi-Level Selection (MLS) Public-Goods World
+
+To probe emergence in a more controlled way, you introduced a **multi-level
+selection (MLS)** world in `multi_level_selection/`:
+
+- `public_goods_sim.py` defines light-weight genomes (baseline, sensitivity,
+  reciprocity) for ~200 agents split into ~10 groups.
+- `animate_cooperation.py` runs repeated public-goods games and visualizes:
+  - global average contribution (as a proxy for cooperation),
+  - trait diversity (baseline std),
+  - total payoff per group each generation.
+- `MLSResultsLogger` writes CSVs into the same Ray-style results tree used by
+  PPO experiments (e.g. `run_config.json`, `mls_generation_metrics.csv`,
+  `predator_final.csv`).
*** End Patch