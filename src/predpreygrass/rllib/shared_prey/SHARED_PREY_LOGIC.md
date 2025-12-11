# Shared Prey Logic

This document summarizes the cooperative capture logic for the `shared_prey` environment and the interactive scenario inspector.

## Core Mechanics
- **Team capture only:** Predators capture prey cooperatively; Moore neighborhood (Chebyshev ≤ 1) defines helper eligibility.
- **Energy threshold:** Helpers’ energies are summed and compared to `prey_energy + team_capture_margin`. If below threshold, prey survives.
- **Energy split:** On success, prey energy is divided equally among helpers and added to their energy.
- **Sequential prey processing:** Prey are handled in deterministic order each step. Outcomes can affect later checks in the same step.

## One-kill-per-step constraint (new)
- **Rule:** A predator that has already eaten earlier in the same step cannot assist in another capture during that step.
- **Implementation:** Helpers list excludes any predator present in `agents_just_ate` before thresholding.
- **Effect:** Prevents cascading multi-prey captures by a single predator within one environment step; inspector reflects this via the real env.

## Inspector Alignment
- The scenario inspector instantiates the real `PredPreyGrass` env for engagement checks, so logic changes (including one-kill-per-step) are reflected automatically.

## Scenario Outcomes
- **No adjacent predators:** Prey survives; no energy changes.
- **Adjacent predators below threshold:** If `sum(helper_energy) < prey_energy + margin`, prey survives; helpers unchanged.
- **Single adjacent predator, above threshold:** Captures the prey; predator gains full prey energy.
- **Multiple adjacent predators, above threshold:** Prey is captured; prey energy is split equally among helpers; helpers gain the split.
- **Multiple prey near one predator:** Prey are processed sequentially; once a predator eats, it is barred from further captures that step, so only the first eligible prey can be eaten.
- **Multiple predators, multiple prey:** For each prey, eligible helpers are those within Chebyshev ≤ 1 that have not yet eaten this step; captures follow the same threshold and equal-split rules per prey.
