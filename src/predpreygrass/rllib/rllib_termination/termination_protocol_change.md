# Termination Protocol Change Notes

## Purpose
- Document the behavioral difference between historical agent-slot reuse and the corrected termination handling.
- Provide guidance for interpreting metrics and evaluating experiments that span the two implementations.

## Terminology
- **Agent slot**: fixed identifier (e.g., `type_1_predator_0`) allocated to an active agent instance.
- **Lifetime**: period between an agent spawning and being marked terminated/truncated.
- **Episode return**: sum of rewards accrued during a single lifetime; reset when a termination signal is emitted.

## Historical Implementation ("Old" Approach)
- Agent slots were recycled without emitting `terminated=True` when a creature despawned.
- RLlib kept an episode open across multiple lifetimes that re-used the same slot.
- Episode returns accumulated rewards across reincarnations, inflating metrics.
- Training logs reported long episode lengths and high totals (e.g., predator return ~100) even when individuals only reproduced a few times per lifetime.
- Credit assignment for learners was noisy because rewards from new lifetimes were credited to prior hidden states.
- num_possible_agents can remain limited to the possible maximum number of active agents at one point in time, without bumping into capacity constraints in the spawning of "new" agents since "died" agents earlier on in the episode can be reused.

## Corrected Implementation ("New" Approach)
- Every lifetime ends with an explicit `terminated=True` or `truncated=True` signal before a slot is reused.
- RLlib resets episode accounting when a creature despawns; new spawns start fresh episodes.
- Episode returns now reflect rewards earned within a single lifetime (e.g., reproduction reward of 10 → return spikes proportional to average births per predator).
- Metrics align with true per-agent performance, improving credit assignment and reproducibility.
- Debug invariants guard against accidental slot reuse without termination.
- Every new spawned agent cannot have been active in the past. If it otherwise should then, after flagged terminations=True, it will produce an error at revivaval, since SingleAgentEpisode.done=True cannot be undone per RLlib protocol. This means num_possible_agents must be as large as the total number of agents ever existed during an episode, in order to not run into capcity constraints of newly spawned agents.

## Observable Differences
- **Episode Return Curves**: values dropped to realistic ranges (e.g., predators near 40 ≈ 4 reproductions per lifetime) versus inflated historical peaks (≈100) caused by reward carry-over.
- **Episode Length Metrics**: average lengths no longer grow indefinitely; they track actual lifetime durations in steps.
- **Learner Stability**: reduced variance in policy updates because hidden states are reset when lifetimes end.
- **Logging**: terminated agents appear in final observation batches, eliminating "acted then truncated" warnings.

## What Did *Not* Change
- Environment dynamics, reward magnitudes, and visual behavior in evaluation grids remain the same.
- Reproduction probability, energy budgets, and other config parameters were untouched; gameplay still matches prior intuition.

## Implications for Experiment Tracking
- Compare historical runs to new ones with caution: legacy metrics overstate per-episode returns and lengths.
- When reproducing published results that relied on the old protocol, note the bookkeeping bug and prefer reruns with the fix for accuracy.
- Hyperparameter searches using the new implementation yield more trustworthy objective signals (e.g., `score_pred`).

## Recommendations
- Treat old logs as qualitatively informative but quantitatively inflated; annotate analyses that included pre-fix data.
- For fair comparisons, rerun key baselines under the corrected termination protocol.
- Keep the debug invariants enabled when extending the environment to catch future regressions in slot management.
