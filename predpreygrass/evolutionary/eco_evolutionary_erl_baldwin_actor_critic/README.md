# ERL Baldwin with linear actor-critic

This experimental variant preserves the baseline ecology, genome, and
evolution, but replaces the immediate evaluation-difference policy update with
a one-step linear TD(0) actor-critic. The inherited action parameters initialize
a private live actor. Every newborn receives a zero-initialized private critic;
neither learned actor changes nor critic state is inherited.

The critic supplies `delta = r + gamma * V(next_obs) - V(obs)`, where
`r = E(next_obs) - E(obs)`. The actor uses the baseline softmax score gradient
scaled by that TD error. Explicit actor and critic update-norm caps prevent a
single transition from producing an unbounded parameter jump. Discounting a
difference reward changes its scientific objective, so gamma is calibrated as
an experimental factor rather than assumed to be innocuous.

This remains in the policy-gradient family and is therefore a diagnostic bridge,
not an algorithmically independent alternative like SARSA.
