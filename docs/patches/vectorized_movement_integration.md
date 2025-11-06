# Vectorized movement (no-LOS) integration guide

This guide shows how to use the optional vectorized movement path without modifying semantics, only when `respect_los_for_movement == False`.

The implementation lives in:

- `src/predpreygrass/rllib/walls_occlusion_simplified/utils/vectorized_movement.py`

It replaces only the hot path of `_process_agent_movements` under the no‑LOS setting. When LOS is on, keep your current per-agent logic.

## Why use it?

- Fewer Python loops and dict lookups in the movement step
- Batch writes to the grid via numpy advanced indexing
- Same rules as your current no‑LOS path (bounds, wall blocking, same-type occupancy blocking; no corner-cut/LOS checks)

## Minimal wiring (opt-in)

Add the import near your other utils in your env module:

```python
from predpreygrass.rllib.walls_occlusion_simplified.utils.vectorized_movement import (
    process_agent_movements_vectorized_no_los,
)
```

Then, in your `_process_agent_movements` method, early-return into the vectorized path when LOS is disabled:

```python
def _process_agent_movements(self, action_dict):
    if not self.respect_los_for_movement:
        process_agent_movements_vectorized_no_los(self, action_dict)
        return
    # existing per-agent implementation continues below for LOS-enabled
    ...
```

No other changes are required. The helper updates:

- `agent_positions`, `agent_energies`, `_per_agent_step_deltas`
- `unique_agent_stats` (distance_traveled, energy_spent, avg_energy_*)
- `grid_world_state` layers for predators (`[1, ...]`) and prey (`[2, ...]`)
- Optional per-agent movement logs via `self._log`

## Assumptions and parity

- Wall channel is `grid_world_state[0]` with 1.0 at wall cells
- Predator layer is `grid_world_state[1]`, prey layer `grid_world_state[2]`
- Action maps are square with side `type_X_action_range` and encoded identical to `_generate_action_map`
- Movement energy cost uses your current rule:
  `distance * move_energy_cost_factor * current_energy`

If LOS movement is enabled, keep the original scalar `_get_move` path to preserve corner-cut and Bresenham line-of-sight checks.

## Notes

- Action maps are cached lazily on the env instance (`_action_map_np_type_1/_type_2`).
- Occupancy blocking uses the pre-move grid layers (same-type only), matching current behavior.
- For many agents per step, this reduces Python overhead and speeds up the movement phase.

---

If you want a fully vectorized LOS-aware path in the future, we can JIT the Bresenham check (e.g., with Numba) and extend this helper while preserving exact semantics.
