# Mammoths: cooperative hunting (shared prey)

## Environment and logic (full description)

### Entities and roles
- **Predators (humans / hunters)<img src="../../../../assets/images/icons/male.png" alt="predator icon" height="
  36" style="vertical-align: middle;">**: `type_1_predator` agents that move, lose energy, hunt, and reproduce.
- **Prey (mammoths)<img src="../../../../assets/images/icons/mammoth_2.jpeg" alt="predator icon" height="
  36" style="vertical-align: middle;">**: `type_1_prey` agents that move, lose energy, eat grass, and reproduce.
- **Grass**: static resource patches that regrow energy over time.
- **Walls** (optional): impassable cells that can be manually placed.

### Grid, observations, and visibility
- The world is a 2D grid (`grid_size` x `grid_size`).
- Each agent observes a local square window around itself (Moore neighborhood), with separate ranges for predators and prey (`predator_obs_range`, `prey_obs_range`).
- Observations include channels for predators, prey, and grass energy; optional line-of-sight masking can hide entities behind walls.

### Startup
- At startup, humans, mammoths, and grass are randomly positioned on the gridworld. Walls surround the gridworld and can optionally be placed within it.

### Actions and movement
- Each agent selects a movement action mapped to a displacement in its Moore neighborhood.
- Predators cannot share a cell with other predators, and prey cannot share a cell with other prey.
- Movement into wall cells is blocked.

### Energy, decay, and death
- All agents have an energy state. Each step, energy decays by a fixed per-step amount (`energy_loss_per_step_predator`, `energy_loss_per_step_prey`).
- If an agent's energy reaches 0 or below, it dies and is removed from the gridworld.

### Foraging and hunting dynamics
- **Prey grazing**: when a mammoth lands on grass, it consumes grass energy and gains that energy (grass energy decreases and can regrow later).
- **Predator hunting**: humans can hunt for mammoths to replenish their energy. They only succeed in hunting and eating if the cumulative energy of humans in the mammoth's Moore neighborhood is greater than the mammoth's energy. On success, the mammoth is removed and its energy is divided among helpers (proportional by default or equal split with `team_capture_equal_split = True`).
- On failure, the mammoth survives and helpers lose energy equal to `E_prey * energy_percentage_loss_per_failed_attacked_prey`, split proportional to their energy share.

#### 1. Proportional Split (default)

By default, the prey’s energy is divided **proportionally to the current energy of each participating predator**:

$$
\Delta E_i = E_{\text{prey}} \cdot \frac{E_i}{\sum_j E_j}
$$

**Implications:**

* Predators with higher energy receive a larger share of the prey.
* Contribution is implicit: bringing more energy to the coalition yields a higher payoff.
* Cooperation is encouraged **only when necessary** (i.e. when no single predator can meet the capture threshold alone).
* This rule tends to produce **hierarchical cooperation**:

  * strong predators dominate kills,
  * weaker predators may trail or be excluded,
  * “rich-get-richer” dynamics can emerge.

**Interpretation:**
This split is purely local and does not require counterfactual reasoning or centralized credit assignment. Cooperation emerges as a *means* to enable capture, not as a rewarded objective.

---

#### 2. Equal Split (optional)

When `team_capture_equal_split = True`, the prey’s energy is divided **equally among all participating predators**:

$$
\Delta E_i = \frac{E_{\text{prey}}}{|\text{helpers}|}
$$

**Implications:**

* All helpers receive the same payoff regardless of their individual energy.
* Low-energy predators are incentivized to stay close to others, as participation alone guarantees reward.
* This often leads to **increased spatial clustering and pack-like movement**.
* However, it also introduces **free-rider incentives**:

  * predators may join late or contribute little energy while still receiving an equal share.

**Interpretation:**
Equal splitting removes implicit hierarchy and favors inclusive cooperation, but at the cost of increased exploitation pressure. Whether stable cooperation emerges becomes a learning problem rather than a structural guarantee.

---

### Comparison Summary

| Aspect            | Proportional Split       | Equal Split               |
| ----------------- | ------------------------ | ------------------------- |
| Reward basis      | Current energy           | Presence                  |
| Cooperation style | Hierarchical / selective | Inclusive / pack-oriented |
| Free-riding       | Discouraged              | Encouraged                |
| Spatial behavior  | Looser coordination      | Stronger clustering       |
| Assumptions       | Minimal, local           | Minimal, local            |
| Credit assignment | Implicit                 | Uniform                   |


### Reproduction
- Humans reproduce asexually when their energy exceeds `predator_creation_energy_threshold`.
- Mammoths reproduce asexually when their energy exceeds `prey_creation_energy_threshold`.
- A child is spawned in a nearby free cell and its starting energy is deducted from the parent.

### Rewards and termination
- Rewards are sparse: agents receive reward only on successful reproduction.
- There is **no explicit cooperation reward**; cooperative hunting emerges from the capture rules and energy accounting.
- An episode ends when either humans or mammoths go extinct, or when `max_steps` is reached.

---

### Research relevance

Both division rules (equal or proportional split) preserve the **minimal-assumption philosophy** of the environment:

* No explicit cooperation reward
* No kin selection or shared team reward
* No centralized or counterfactual credit assignment

The difference lies in **what kind of cooperation agents are able to learn**:

* proportional split emphasizes *power-based coalition formation*,
* equal split emphasizes *presence-based cooperation and coordination*.

Comparing these regimes allows us to study how reward division alone shapes emergent cooperative behavior in multi-agent reinforcement learning.


# MADRL training

- Predators and Prey are independently (decentralized) trained via their own RLlib policy module.

  - **Predator** 
  - **Prey**

  - Predators and Prey **learn movement strategies** based on their **partial observations**.

# Results
<p align="center">
    <b>Emerging cooperative hunting in Predator-Prey-Grass environment</b></p>
<p align="center">
    <img align="center" src="./../../../../assets/images/gifs/cooperative_hunting_mammoths_15MB.gif" width="600" height="500" />
</p>

- Cooperative hunting occurs, though it is **not strictly imposed nor rewarded**.
- Human hunters tend to cluster together.
