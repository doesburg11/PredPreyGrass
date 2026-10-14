# Reputation and Conditional Joining

This note describes a light-weight "reputation" signal for predators and how it can
enable conditional cooperation without forcing it. The idea is to *expose* history,
not to impose any new capture rules or hard thresholds.

## 1) Reasons

- Reciprocity without coercion: agents can learn "join with reliable partners"
  while still retaining the option to defect.
- Indirect reciprocity: behavior toward others can depend on third-party history,
  enabling richer social dynamics than one-shot join/defect.
- Long-horizon credit assignment: short-term gains from defection can be weighed
  against future access to cooperative captures.
- Minimal disruption: keep the same capture rules and energy mechanics, only add
  a new observation signal and optional metrics.

## 2) Tweaks to current environment

Keep these changes optional with defaults off to avoid breaking existing models.

### A) Track a reputation score (per predator)

Define reputation as a rolling join rate or an EMA of join decisions:

- `rep` in [0, 1], updated each step a predator has an opportunity to join
  (e.g., when at least one prey is in Moore neighborhood).
- Rolling window: `rep = mean(join_hunt in last K opportunities)`
- EMA: `rep = (1 - alpha) * rep + alpha * join_hunt`

Suggested config knobs:

- `reputation_enabled` (bool, default False)
- `reputation_window` (int, default 50) or `reputation_ema_alpha` (float, default 0.1)
- `reputation_opportunity_only` (bool, default True) to avoid counting idle steps
- `reputation_min_samples` (int, default 5) for stability in early steps
- `reputation_noise_std` (float, default 0.0) to model imperfect information

### B) Expose reputation in observations (no rule changes)

Offer one or both observation modes:

- Grid channel: an extra predator channel that stores each predator's reputation
  at their position (neighbors can "see" it within observation range).
- Scalar feature: append own reputation and a summary of nearby reputations
  (mean/min/max of neighbor rep) to the non-spatial observation vector.

Suggested config knobs:

- `include_reputation_channel` (bool, default False)
- `include_reputation_summary` (bool, default False)
- `reputation_visibility_range` (int, default predator_obs_range)

### C) Metrics (analysis only)

Extend custom metrics to track conditional behavior:

- Join rate conditioned on neighbor reputation (e.g., join when mean rep > 0.6)
- Capture success vs partner reputation
- Free-rider exposure vs partner reputation

This gives you evidence of conditional cooperation without hard-coding it.

## 3) Expected outcomes

- Emergent conditional cooperation: predators join more often with high-rep
  neighbors and defect when surrounded by low-rep agents.
- Reputation stratification: "good" joiners cluster; persistent defectors become
  isolated or only benefit from scavenging.
- More stable cooperation: repeated interactions create incentives to maintain
  reputation, improving long-term team captures without forcing them.
- Potential downsides: noisy rep signals can cause mistrust and collapse; early
  bad luck can trap agents in low-rep states unless rep decays or is reset.

