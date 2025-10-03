# Walls & Occlusion Experiments in PredPreyGrass

<p align="center">
    <b>Trained Predator-Prey-Grass walls-occlusion environment</b></p>
<p align="center">
    <img align="center" src="../../../../assets/images/gifs/kin_selection.gif" width="600" height="500" />
</p>

This directory contains experiment scripts, configs, and documentation for studying the effects of static and dynamic walls, as well as line-of-sight (LOS) occlusion, on multi-agent co-evolution in the Predator-Prey-Grass (PPG) environment.

## Overview

In the PPG environment, agents (predators and prey) interact on a gridworld with grass as a renewable resource. Walls can be placed to create obstacles, corridors, or complex mazes. When occlusion is enabled, walls block agents' line of sight, affecting both their observations and their ability to move. This setup allows for the study of:
- How spatial structure and visibility constraints shape co-evolutionary dynamics
- The emergence of new strategies in response to environmental complexity
- The robustness of learned behaviors to changes in wall layout or occlusion rules

## Key Features

- **Manual and Dynamic Wall Placement:**
  - Walls can be placed manually (via config) or moved/reshuffled dynamically during training.
  - Dynamic walls can change every N steps, forcing agents to adapt to a non-stationary environment.
- **Occlusion (LOS Masking):**
  - When enabled, agents' observations are masked by line-of-sight; they cannot see through walls. The Field Of Vision (FOV) in the gridworld is determined by the [*Bresenham's line algorithm*](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)
  - Movement can also be restricted to only those cells visible via LOS.
- **Flexible Experimentation:**
  - Easily switch between static/dynamic walls, with or without occlusion, by changing config flags.
  - Supports a variety of wall layouts: mazes, chambers, forests, arenas, etc.

## How It Works

- **Configuring Walls:**
  - Set `wall_placement_mode` to `manual` and provide a list of `(x, y)` coordinates in `manual_wall_positions` for static layouts.
  - For dynamic walls, set `dynamic_walls=True` and specify `dynamic_wall_update_freq` (steps between wall updates).
- **Enabling Occlusion:**
  - Set `mask_observation_with_visibility=True` to mask observations by LOS.
  - Set `respect_los_for_movement=True` to restrict movement to LOS-visible cells.
  - Optionally, set `include_visibility_channel=True` to add a visibility mask as an extra observation channel.
- **Running Experiments:**
  - Use the provided `tune_ppo_kin_selection.py` (or similar) script to launch training with your chosen wall/occlusion settings.
  - Evaluation scripts (e.g., `evaluate_ppo_from_checkpoint_debug.py`) visualize agent behavior and wall/LOS overlays.

## Example Config Snippet

```python
config_env = {
    # ... other env settings ...
    "wall_placement_mode": "manual",
    "manual_wall_positions": [(6,6), (7,6), ...],
    "dynamic_walls": True,
    "dynamic_wall_update_freq": 50,  # Move walls every 50 steps
    "mask_observation_with_visibility": True,
    "respect_los_for_movement": True,
    "include_visibility_channel": True,
}
```

## Suggested Wall Layouts
- **Central Maze:** Spiral or labyrinth in the center.
- **Chambers:** Multiple rooms with narrow corridors.
- **Forest:** Scattered single-tile obstacles.
- **Arena:** Perimeter walls with a few gates.
- **Dynamic:** Walls move or change during training.

## Research Questions
- How do static vs. dynamic walls affect predator/prey strategies?
- Does occlusion promote more robust or generalizable behaviors?
- Can agents adapt to non-stationary environments with moving obstacles?

## Usage
1. Edit the config file to set your desired wall and occlusion parameters.
2. Run the training script:
   ```bash
   python -u src/predpreygrass/rllib/ppg_visibility/tune_ppo_kin_selection.py | tee logs/last_run_tune.log
   ```
3. Visualize and evaluate results using the provided evaluation scripts.

## Files
- `tune_ppo_kin_selection.py`: Main training script for wall/occlusion experiments.
- `evaluate_ppo_from_checkpoint_debug.py`: Visual evaluation with wall and LOS overlays.
- `config_env_train_2_policies.py`: Example config with manual/dynamic wall options.
- `README.md`: This documentation.

## Tips
- For reproducibility, always snapshot your config and wall layout with each experiment.
- Try both static and dynamic walls to see how agent strategies change.
- Use the visibility channel for richer policy learning, or mask observations for harder partial observability.

---

For more details, see the main project README and the code comments in each script.
