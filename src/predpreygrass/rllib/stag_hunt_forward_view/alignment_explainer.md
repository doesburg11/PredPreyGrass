**Facing/Alignment Explainer**
This note explains how the “deliberate vs coincidental” label is computed for predator attack attempts in `extract_predator_trajectory.py`.

**Facing**
- `facing_x` / `facing_y` are the predator’s last non-zero movement direction (normalized to -1, 0, 1).
- If the predator does not move on a step, facing stays at its previous non-zero direction.

**Alignment Computation (per event step)**
- For each step that has `eat` or `failed` events, we look up predator and prey positions from `per_step_agent_data_{run}.json`.
- We compute the vector from predator to prey and compare it to facing:

```text
dx = prey_x - predator_x
dy = prey_y - predator_y
distance = sqrt(dx^2 + dy^2)
dot = dx * facing_x + dy * facing_y
angle = acos(dot / (distance * |facing|))   # in degrees
```

**Labels**
- `alignment_prey_in_front` is true when `dot > 0` (prey in front half-plane).
- `alignment_deliberate` is true when `distance <= 3.0` and `angle <= 45°`.
- If multiple prey are involved in the same step, we use the closest one for the alignment fields.

**Visual**
Legend: P = predator, R = prey, arrow = facing.

![Deliberate vs coincidental example](../../../../assets/eval_comparison_summary_plots/deliberate_vs_coincidental.svg)

**Caveat**
Facing reflects the last non-zero movement, not necessarily movement in the current step. If the predator is stationary, the arrow may be stale.
