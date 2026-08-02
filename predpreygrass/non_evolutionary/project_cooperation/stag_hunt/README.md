# "Stag hunt": cooperative and solo hunting with large and small prey

## Overview

- The environment is a gridworld with Predators, Prey, and Grass.
- Predators come in one type:
  - Humans <img src="../../../../assets/images/icons/male.png" alt="predator icon" height="
  36" style="vertical-align: middle;"> (type_1_predator)
- Prey come in two types:
  - Large prey; Mammoths <img src="../../../../assets/images/icons/mammoth_2.jpeg" alt="predator icon" height="
  36" style="vertical-align: middle;"> (type_1_prey)
  - Small prey; Rabbits <img src="../../../../assets/images/icons/prey.png" alt="predator icon" height="
  36" style="vertical-align: middle;"> (type_2_prey)
- Walls surround the grid and can be manually placed inside the grid if desired.

## Movement and occupancy

- Agents move in a Moore neighborhood (8 directions plus stay).
- Predators cannot share a cell with other predators.
- Prey cannot share a cell with other prey.
- Agents cannot move into Wall cells.

## Energy, death, and grass

- Predators and prey lose energy every step.
- If an agent's energy drops to 0 or below, it dies and is removed.
- Prey eat grass by landing on a grass cell. Grass energy is reduced by the prey's bite size (clamped to 0).
- Rabbits have a smaller bite size; Mammoths have a larger bite size.
- Mammoth feeding leaves at least the rabbit bite size in the patch so rabbits can still feed.

## Predation and cooperative capture

- Predators (humans) attempt capture for any prey in their Moore neighborhood.
- A prey is captured if the cumulative predator energy in its Moore neighborhood is larger than the prey's own energy.
- Rabbits are low-energy and can usually be captured by a single predator.
- Mammoths are high-energy and therefore typically require multiple cooperative predators.
- Failed capture applies a struggle penalty: total penalty is
  `prey_energy * energy_percentage_loss_per_failed_attacked_prey`, split proportionally across attackers.
- Optionally, predators are getting killed in a failed attempt to kill prey.
- Successful capture removes the prey and redistributes its energy among attackers:
  - proportional split by predator energy (default), or
  - equal split when `team_capture_equal_split = True`.

## Reproduction and rewards

- Predators and prey reproduce asexually when they reach their type-specific energy threshold.
- Offspring spawn in the Moore neighborhood; the parent pays the offspring's initial energy.
- Rewards are sparse: agents are only rewarded on reproduction. Eating affects energy, not reward.
- Optionally, a negative reward can be implemented for agents dying.

## Episode end

- The episode ends when all predators (humans) are extinct, or all prey (mammoths + rabbits) are extinct,
  or when `max_steps` is reached.

# MADRL training

- Humans (predators), mammoths (prey) and rabbits (prey) are as groups independently (decentralized) trained via their own RLlib policy module.
- All three agent types learn movement strategies based on partial observations of the full state (gridworld).

# Results

- Cooperative hunting occurs, though it is not explicitly rewarded.
