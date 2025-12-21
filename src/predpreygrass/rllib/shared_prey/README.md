# Cooperative Hunting: Features "Shared Prey" environment

- At startup Predators <img src="../../../../assets/images/icons/predator.png" alt="predator icon" height="
  24" style="vertical-align: middle;">, Prey <img src="../../../../assets/images/icons/prey.png" alt="prey icon" height="
  24" style="vertical-align: middle;"> and Grass are randomly positioned on the gridworld. Walls are surrounding the gridworld and are possibly manually placed within the gridworld. 

- Predators and Prey can move in a Moore neighborhood. Predators cannot share a cell with other Predators. Prey cannot share a cell with other Prey. Predator nor Prey can move to a Wall cell.

- Predators and Prey possses energy which depletes every time step.

- If the energy of an agent is zero, the agent dies and is removed from the gridworld.

- Predator can eat Prey and Prey can eat grass to replenish their energy.

- **Cooperative capture and energy accounting**: A Prey is eaten by (a) Predator(s) if the cumulative energy of all Predators in its Moore neighborhood is *strictly larger* than the Prey's own energy (optionally with a safety margin via `team_capture_margin`). This holds for a single Predator as well as a group.
  - **Failed attempt (struggle cost)**: If the cumulative energy is too low, the attack fails and the Prey survives. The attacking Predators lose energy to reflect the struggle. The total penalty is `prey_energy * energy_percentage_loss_per_failed_attacked_prey` and is **subtracted proportionally** from each attacker based on their energy share. This applies to solo and team attempts. If the penalty drops a Predator to zero or below, it dies that step.
  - **Successful capture (no struggle cost)**: When the attackers have higher cumulative energy, the Prey is removed and **no extra energy loss** is applied to Predators. The Prey's energy is then split among attackers (proportional by default, or **equal split** when `team_capture_equal_split = True`).

- A Grass patch is eaten if a Prey lands on its cell.

- If a Grass patch is eaten, its energy is set to zero and the corresponding Prey receives its energy.

- Grass gradually regenerates at the same spot after being eaten by Prey. Grass, as a non-learning agent, is being regarded by the model as part of the environment, not as an actor.

- When a Predator or Prey reaches a certain energy treshold by eating, it asexually reproduces. Its child is placed in the Moore neighborhood of its parent. The initial energy of a child is deducted from the parent.

- Agents only receive a reward when they reproduce. All other behavior is emergent. The sparse rewards configuration shows that the ecological system is able to sustain with this minimalistic optimized incentive for both Predators and Prey.

- The game ends when either the number of Predators or the number of Prey is zero.


# MADRL training

- Predators and Prey are independently (decentralized) trained via their own RLlib policy module.:

  - **Predator** 
  - **Prey**

  - Predators and Prey **learn movement strategies** based on their **partial observations**.

# Results
<p align="center">
    <b>Emerging cooperative hunting in Predator-Prey-Grass environment</b></p>
<p align="center">
    <img align="center" src="./../../../../assets/images/gifs/cooperative_hunting_9MB.gif" width="600" height="500" />
</p>

- Cooperative hunting occurs, though it is **not strictly imposed nor rewarded**.
