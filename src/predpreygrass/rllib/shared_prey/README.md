# Shared-prey cooperative capture (Moore neighborhood)

> Note: lineage survival rewards are **disabled** in `shared_prey`; only capture/consumption/reproduction rewards remain active. The lineage-focused documentation below is retained for historical reference but not active in this environment.

- Predators can capture a prey only when the **sum of predator energies in the prey's Moore neighborhood (radius 1, including diagonals/center)** meets or exceeds the prey's energy (optional margin `team_capture_margin`).
- On a successful capture, prey energy intake and the predator catch reward are split **equally** across the contributing predators (`team_capture_equal_split=True`).
- Toggle via `team_capture_enabled` in `config/config_env_shared_prey.py`; summary counters land in `infos['__all__']` (`team_capture_successes`, `team_capture_failures`, `team_capture_last_helpers`).
- Predators blocked by the carcass-only window are ignored for live-prey captures; they still receive step reward only.

# Lineage Rewards & Fertility Caps in `lineage_rewards`

This note explains why `lineage_rewards` replaces the legacy **kin kick-back** rewards with **lineage-based survival rewards**, how the new mechanism is implemented, and how recently added **fertility-age caps** interact with the limited-intake energy system to shape kin-directed (altruistic) behavior.

> **Scope**
> - Target environment: `src/predpreygrass/rllib/lineage_rewards/predpreygrass_rllib_env.py`
> - Evaluation helpers: `evaluate_ppo_from_checkpoint_*.py`
> - Config knobs: `config/config_env_lineage_rewards.py`

---

## 1. Motivation: from Kin Kick-Backs → Lineage Survival Rewards

| Aspect | `kick_back_rewards` (legacy) | `lineage_rewards` lineage logic |
| --- | --- | --- |
| Reward trigger | Immediate reproduction event; parent (and in some setups grandparent) gets a one-time kick-back proportional to offspring type. | Every step where the **number of living descendants increases** produces a reward for every ancestor still alive. |
| Information needed | Parent ID only. | Parent chain (parent, grandparent, ...), live descendant counts. |
| Resilience to identity reuse | Fragile (ID reuse per episode corrupts ancestry). | Works with **never-reuse** agent IDs; lineage tracker mirrors true family trees. |
| Behavioral signal | “Spam births” as fast as energy allows. Survival of offspring is irrelevant once the kick-back lands. | “Keep my descendants alive” because rewards accumulate while descendants remain alive (and only while ancestor is alive). |
| Logging | `kin_kickbacks` counter per agent. | `lineage_reward_total`, per-step `reward_events`, ancestor ↔ descendant event log. |

### Why change?
1. **Reward timing** – Under kick-backs, evolution optimizes only for raw reproduction rate. Agents dump offspring even in hostile territory because survival has no bearing on reward.
2. **Slot reuse** – Kick-backs break when IDs are reused mid-episode; the ancestor relationship collapses. `lineage_rewards` never reuses IDs within an episode, making lineage math tractable.
3. **Credit assignment** – Survival-based lineage rewards deliver a smoother signal tied to colony health, producing more stable PPO objectives (`score_pred`).

Mathematically, the intended signal mirrors the original lineage design note:

```text
R_lineage(i, t) = λ * (L_i(t) – L_i(t-1))
```

`L_i(t)` counts how many *living* descendants agent `i` has at step `t`. Positive deltas reward net growth, negative deltas could penalize net loss. The current `lineage_rewards` code awards only the positive portion (no penalty when descendants die), but the bookkeeping keeps both `live_descendants` and `prev_live_descendants`, so enabling symmetric rewards would be straightforward if future experiments call for it.

---

## 2. Lineage Reward Implementation

### 2.1 Data structures
- `self.lineage_tracker[agent_id] = {parent_id, children_ids, live_descendants, prev_live_descendants, is_alive_descendant}`
- `self.agent_stats_live[agent_id]['lineage_reward_total']` aggregates payouts for analysis/evaluation.
- `self.agent_event_log[agent_id]['reward_events']` records each lineage payout with timestamp and cumulative reward.

### 2.2 Lifecycle hooks
1. **Birth** (`_register_new_agent` → `_handle_lineage_birth`)
   - Creates tracker entry, links child to parent, marks the new agent as “alive descendant”.
   - `live_descendants` counters for all ancestors increment by 1.
2. **Death** (`_handle_lineage_death`)
   - Clears the alive flag, decreases ancestor live-descendant counters.
3. **Update** (`_apply_lineage_survival_rewards`)
   ```python
   delta = record['live_descendants'] - record['prev_live_descendants']
   reward = lineage_reward_coeff[type_policy] * delta  # only positive delta pays out
   ```
   - Reward deposited into `self.rewards`, `agent_stats_live`, debug counters, and event log.
   - Negative deltas (descendants dying) currently yield no penalty.

### 2.3 Config surface (`config_env_lineage_rewards.py`)
```python
"lineage_reward_coeff": {
    "type_1_predator": 0.35,
    "type_1_prey": 0.60,
    # other policy groups set to 0.0 by default
}
```
- Coefficients scale the per-descendant reward. Setting them to `0.0` reverts to reproduction-only incentives.

### 2.4 Evaluation outputs
- `agent_event_log_*.json`: includes per-agent reward events so you can trace which ancestors benefited.
- `reward_summary.txt` and console fitness summaries now print `Lineage` columns per agent group after the evaluator changes in this branch.

### 2.5 Implementation sketch (from the legacy lineage note)
The helper methods `_handle_lineage_birth`, `_handle_lineage_death`, and `_propagate_lineage_delta` directly mirror the first Copilot design draft:

```python
def _register_birth(child_id, parent_id):
   self.lineage_tracker[child_id]["parent_id"] = parent_id
   self.lineage_tracker[parent_id]["children_ids"].add(child_id)
   ancestor = parent_id
   while ancestor is not None:
      self.lineage_tracker[ancestor]["live_descendants"] += 1
      ancestor = self.lineage_tracker[ancestor]["parent_id"]

def _register_death(agent_id):
   ancestor = self.lineage_tracker[agent_id]["parent_id"]
   while ancestor is not None:
      self.lineage_tracker[ancestor]["live_descendants"] -= 1
      ancestor = self.lineage_tracker[ancestor]["parent_id"]
```

`_apply_lineage_survival_rewards` simply compares the live/prev-live counters to compute the `R_lineage` term above. Keeping this pseudo-code in the doc helps anyone cross-referencing the older `lineage_rewards/lineage_reward_notes.md` file understand how the final implementation maps to the prototype.

---

## 3. Fertility & Lifespan Caps

### 3.1 Fertility configuration
```python
"max_fertility_age": {
    "type_1_predator": 160,   # example
    "type_1_prey": 120,
    # set to None for unlimited fertility (default)
}
```
- Values count **environment steps** since birth.
- `None` (or negative) = unlimited fertility (legacy behavior).

### 3.2 Runtime behavior (fertility)
1. During `_apply_time_step_update`, agent age increments and `_maybe_mark_fertility_expired` stamps the first step at which age ≥ cap.
2. `_agent_is_fertile(agent_id)` guards `_handle_*_reproduction`.
3. When an infertile agent tries to reproduce:
   - Reproduction is skipped without energy transfer or reward.
   - Counters `reproduction_blocked_due_to_fertility_{predator,prey}` increment.
   - `agent_stats_live` logs `fertility_blocked_attempts` and `fertility_expired_step`.
   - Event log receives a `fertility_events` entry; evaluators bubble these stats to the console/CSV.

### 3.3 Fertility rationale & effect
- After fertility expires, the **only** remaining reward channel for that agent is lineage payouts. This creates a clear late-life incentive to guard/feed offspring rather than chasing new births that are now impossible.
- The cap also mirrors biological realities (senescence) and prevents immortal founders from hoarding reproduction rewards indefinitely.

### 3.4 Lifespan configuration
```python
"max_agent_age": {
   "type_1_predator": 220,
   "type_2_predator": None,
   "type_1_prey": 180,
   "type_2_prey": None,
}
```
- Values count **steps lived**. `None` (or negative) keeps the legacy immortal behavior.
- Defaults mirror the `config_env_lineage_rewards.py` sample and can be tuned per policy group just like fertility caps.

### 3.5 Runtime behavior (max age)
1. `_apply_time_step_update` increments `agent_ages` for every non-carcass agent and calls `_agent_age_exceeded` to detect caps.
2. Agents that reach the limit are routed through `_terminate_agent_due_to_age`, which:
   - Removes them from the grid and active lists while calling `_handle_lineage_death` so ancestor counts stay accurate.
   - Marks `death_cause="max_age"`, stamps `age_expired_step`, and emits an info flag `terminated_due_to_age`.
   - Appends a `lifecycle_events` entry so evaluators can audit which agents timed out.
3. Because age-outs happen before engagements, follow-up combat/feeding logic automatically skips the retired agent in the same step.

### 3.6 Why limit lifetimes?
- Prevents immortal founders from monopolizing lineage rewards for the entire episode, which previously dampened turnover pressure.
- Forces lineages to refresh, keeping the event logs and reward streams focused on actively reproducing branches rather than indefinitely stable sentinels.
- Pairs with fertility caps: once reproduction is blocked and the age limit approaches, optimal behavior is to protect descendants, not hoard energy.

### 3.7 Juvenile carcass-only diet window
```python
"carcass_only_predator_age": {
   "type_1_predator": 30,
   "type_2_predator": None,
}
```
- Predators younger than the configured step count may only bite carcasses (prey already in the `dead_prey` set). The helper `_predator_requires_carcass_only` enforces this per-policy window.
- When a juvenile lands on a live prey tile, `_handle_predator_engagement` now short-circuits, applies only the idle/step reward, and logs a `diet_events` entry (`carcass_only_block`). No energy is transferred and the prey remains alive.
- `agent_stats_live[*]["carcass_only_blocks"]`, `env.carcass_only_live_prey_blocks_predator`, and the per-step `infos` flag (`carcass_only_live_prey_blocked`) expose how often this throttle fires during training/eval.
- Founders spawned during `reset()` start with their age equal to the configured window so they can hunt immediately; newborn predators still begin at age 0 and must rely on carcasses until they age out.
- Set the value to `None` (or a negative integer) to disable the constraint for a policy group.


---

## 4. Interplay with Limited Intake & Altruism

Limited intake mechanics (predators take multiple small bites; grass regrows slowly under capped energy gain) already force agents to budget food. Adding lineage rewards and fertility caps changes how that budgeting manifests:

1. **Resource sharing** – Parents, especially post-fertility, benefit more from letting juveniles consume carcasses/grass because every additional survivor adds to their lineage reward stream.
2. **Guarding behavior** – Limited intake makes contested carcasses last multiple steps. Ancestors escorting kin can deny opponents those bites, protecting both kin survival and their own future lineage income.
3. **Population stability** – Fertility caps reduce late-episode reproduction bursts. Combined with limited intake, this shifts pressure toward **maintaining** existing offspring rather than producing disposable ones.
4. **Anti-hoarding** – Since an infertile agent can no longer convert personal energy into reproduction reward, the best use of surplus energy is to survive long enough to chaperone kin (keep lineage bonuses flowing) and to physically shield food patches for them.
5. **Juvenile provisioning** – The carcass-only window forces young predators to rely on elders for processed food, magnifying incentives for cooperative carcass sharing and territorial defense around downed prey.

These interactions seed altruistic behaviors (e.g., vacating grass for kin, intercepting predators) without explicitly coding cooperation rules—the incentives emerge from the reward redesign plus energy constraints.

---

## 5. Practical Tips

- **Tuning lineage coeffs**: Start small (0.1–0.5). Too large and PPO gradients over-emphasize survival counts, causing conservative policies that under-hunt/forage.
- **Monitoring**: Watch `lineage_reward_total`, `fertility_expired_step`, and the fertility block counters printed during evaluation to verify the mechanics are active.
- **Capacity planning**: Because IDs are never reused, ensure `n_possible_type_*` exceeds the expected total births per episode; otherwise reproduction will be blocked by capacity before fertility caps matter.
- **Regression safety**: Keep the lineage/fertility event logs when running Tune sweeps; they’re invaluable for diagnosing why a trial stalls (e.g., if everyone hits fertility caps prematurely).

---

## 6. Validation & Regression Tests
- Unit tests for the lineage rewrite live in `tests/test_lineage_rewards_validation.py` and can be run with `pytest -q tests/test_lineage_rewards_validation.py` (last run: 2025-12-04).
- `test_lineage_reward_triggers_on_descendant_gain` programmatically registers a child via `_register_new_agent` to assert the lineage delta pays out and logs an event.
- `test_agent_emits_max_age_termination_and_logs_event` forces an agent to the configured age cap, steps the env once, and ensures the termination path populates infos plus completed stats.
- `test_juvenile_predator_blocked_from_live_prey` and `test_predator_can_eat_live_prey_after_window` validate the carcass-only diet window (blocked bite logging vs. normal catch once the agent ages out).
- `test_founder_predator_starts_at_carcass_threshold` ensures reset-time founders are pre-aged past the juvenile window so the ecosystem begins with active predation.
- Keep these tests in CI to catch regressions whenever lineage bookkeeping or cap logic changes.

## 7. References
- `predpreygrass_rllib_env.py`: `_handle_lineage_birth`, `_apply_lineage_survival_rewards`, `_agent_is_fertile`.
- `evaluate_ppo_from_checkpoint_debug.py`: new lineage + fertility columns in reward summaries.
- Historical behavior snapshot: `src/predpreygrass/rllib/kick_back_rewards/README.md` (original kick-back design).

Feel free to extend this document with empirical findings (plots, lineage depth distributions, etc.) as new experiments roll in.
