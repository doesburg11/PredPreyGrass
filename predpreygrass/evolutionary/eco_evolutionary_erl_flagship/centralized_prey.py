"""A single, shared, GENOME-CONDITIONED prey action policy -- one set of
action_weights/action_bias, pooling experience from every living prey's own
step each generation, rather than each prey learning its own action network
alone within its own (short) lifetime. Mirrors centralized_predator.py's
already-proven pooling pattern, with one essential difference: the predator's
policy is fully shared with NO per-individual reward diversity (there's nothing
for it to express); prey's whole scientific point is that DIFFERENT genomes
should be able to produce DIFFERENT behavior, so the shared network's input is
augmented with the individual's own `eval_weights` alongside the observed
features -- the same weights can still express different policies for
different genomes, the way a single hypernetwork or genome-conditioned policy
does in the wider ML literature (see e.g. Wang et al. 2019's evolved-intrinsic-
motivation architecture, discussed in this module's README.md's status
section -- the direct prior art for this design).

Motivation, not guessed at: polymorphism_check.py found that even with a
strong reward-to-behavior coupling (20x learning rate -- see lr_sweep.py),
Trial 13's default per-individual-learner architecture never lets evolution
discover anything like `avoider`'s extreme, fitness-consequential region of
`eval_weights` space from a neutral start; the population's spread stayed
flat/bounded over ~260 generations. The existing Wang2019 replication
(`/home/doesburg/Projects/Wang2019`) hit an analogous null result for a
directly related architecture, root-caused to each genotype getting only
~20 episodes of inner-loop RL training before being scored -- far too little
to separate real genotype quality from noise. Individual prey lifetimes here
(~100-500 steps, bounded by the ecology's own predation/energy dynamics) are
similarly short for ANY within-lifetime learner. Pooling experience across the
whole live population, the way the predator already does, directly attacks
that volume-of-experience bottleneck without needing a heavier within-lifetime
algorithm (PPO) that would need even MORE steps per trial to converge, not
fewer.

Three real problems found and fixed in sequence, not guessed at once, each
confirmed by direct debugging before moving to the next:

1. A first version of `augment` simply concatenated features and genome.
   FAILED -- not a bug, an architectural dead end: `action_probs`/
   `reinforce_update` (networks.py) are purely linear (`logits = obs @
   weights + bias`), so concatenating genome onto the input can only shift a
   CONSTANT, genome-dependent offset on the logits -- it cannot change HOW
   the network responds to a given feature value depending on genome, since
   a linear map has no interaction terms between input dimensions. Fixed by
   appending the flattened OUTER PRODUCT of features and genome (see
   `augment`) -- a bilinear interaction term, not a nonlinear/hidden-layer
   network. The effective weight applied to feature_i becomes `W[i] +
   sum_j(genome_j * W_interaction[i, j])`, which DOES depend on genome,
   while the update rule stays exactly `reinforce_update`'s existing plain
   outer-product gradient.

2. Even with genuine gating capacity, the population still didn't reach
   stable, healthy sizes (predators died out normally -- confirmed identical
   to the default architecture -- but prey then declined toward near-
   extinction instead of plateauing). Root cause, found by direct
   measurement: with 80 input dimensions (8 features + 8 genome + 64
   interaction) all sharing the SAME weight-init scale as the default
   architecture's 8-dim input, initial logit variance is roughly 20x higher
   -- a far more peaked, less-exploratory starting policy than the
   deliberately near-uniform-random starting point every earlier Trial 13
   result relies on (see README.md's entropy-based predator-transfer
   diagnosis). `init_std` and `interaction_scale` below exist to correct
   this -- calibrated empirically (measure real feature/genome/interaction
   value distributions from a live run, solve for the weight scale that
   reproduces the default architecture's initial logit variance), not
   guessed.

3. Even with (1) and (2) fixed, small/fragile populations persisted. Root
   cause: `update()` was called immediately, once per living prey, every
   step -- meaning up to several dozen full-strength, sequential single-
   example gradient updates land on the SAME shared weights within one env
   step, each one seeing the weights already changed by the previous
   agent's update. That is a much less stable training regime than either
   the default architecture (one private update per individual, zero
   interference) or genuine batched SGD (many examples' gradients computed
   against the SAME frozen weights, then averaged into one step). Fixed by
   `accumulate()`/`apply_batch()`: every living prey's (obs, action,
   reinforcement) this step is buffered, then ONE averaged update is
   applied at the end of the step, every example's gradient computed
   against the same pre-batch weights -- see driver.py's `step()`.
"""

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.networks import (
    action_probs,
    sample_action,
)


class CentralizedPreyPolicy:
    """`obs_dim` is the raw feature count (8); the shared network's actual input
    is `obs_dim + genome_dim + obs_dim*genome_dim` (features, genome, and their
    flattened, rescaled outer product -- see `augment`), so `genome_dim` must
    be passed too. `interaction_scale` and `init_std` should be calibrated to
    the actual feature/genome value distributions in use (see module
    docstring, point 2) -- the defaults below match this module's own
    features.py/config.py at the time of calibration, not universal
    constants."""

    def __init__(
        self,
        obs_dim: int,
        genome_dim: int,
        n_actions: int,
        rng: np.random.Generator,
        init_std: float = 0.107,
        lr_positive: float = 0.05,
        lr_negative: float = 0.02,
        interaction_scale: float = 2.5,
    ):
        self.input_dim = obs_dim + genome_dim + obs_dim * genome_dim
        self.action_weights = rng.normal(0.0, init_std, size=(self.input_dim, n_actions))
        self.action_bias = rng.normal(0.0, init_std, size=n_actions)
        self.lr_positive = lr_positive
        self.lr_negative = lr_negative
        self.interaction_scale = interaction_scale
        self._pending: list[tuple[np.ndarray, int, float]] = []
        self._n_zero = 0

    def augment(self, features: np.ndarray, eval_weights: np.ndarray) -> np.ndarray:
        """The shared network's actual input: features, genome, and their
        flattened outer product (a bilinear interaction term, rescaled by
        `interaction_scale` to compensate for its naturally smaller per-entry
        magnitude -- see module docstring, point 2)."""
        interaction = np.outer(features, eval_weights).ravel() * self.interaction_scale
        return np.concatenate([features, eval_weights, interaction])

    def act(self, augmented_obs: np.ndarray, rng: np.random.Generator) -> int:
        probs = action_probs(augmented_obs, self.action_weights, self.action_bias)
        return sample_action(probs, rng)

    def accumulate(self, augmented_obs: np.ndarray, action: int, reinforcement: float) -> None:
        """Buffer one individual's (obs, action, reinforcement) for this step's
        batched update -- does NOT touch the shared weights. Called once per
        living prey per step (driver.py's `_select_prey_action`); `apply_batch`
        applies everything buffered so far in one averaged step, see
        driver.py's `step()`. Zero-reinforcement examples are still counted
        (`_n_zero`) even though they contribute no gradient (matching
        reinforce_update's original no-op) -- a Codex review caught that
        silently dropping them from the batch denominator would have made
        the averaged step's effective size depend on how many of this step's
        living prey happened to have exactly-zero reinforcement, rather than
        the batch size actually documented. In practice this is a real fix,
        not just a theoretical one for the future: it matters whenever
        reinforcement can legitimately be exactly 0.0 (e.g. sparse-reward
        prey), even though for this architecture's default genome-driven
        `e_now - prev_eval` signal it's a continuous quantity that's
        essentially never exactly zero (confirmed directly: 0/32,901 calls
        in a real run)."""
        if reinforcement == 0.0:
            self._n_zero += 1
            return
        self._pending.append((augmented_obs, action, reinforcement))

    def apply_batch(self) -> None:
        """Apply ONE averaged update from every buffered example this step, each
        gradient computed against the SAME (pre-batch) weights -- standard
        batched-gradient stability, unlike applying every example's update
        immediately and sequentially (module docstring, point 3). The
        denominator is every example seen this step (nonzero AND zero
        reinforcement), not just the nonzero ones -- see `accumulate`."""
        n = len(self._pending) + self._n_zero
        if n == 0:
            return
        total_weight_step = np.zeros_like(self.action_weights)
        total_bias_step = np.zeros_like(self.action_bias)
        for obs, action, reinforcement in self._pending:
            probs = action_probs(obs, self.action_weights, self.action_bias)
            grad_logits = -probs
            grad_logits[action] += 1.0  # d log pi(a|obs) / d logits
            lr = self.lr_positive if reinforcement > 0 else self.lr_negative
            step = lr * reinforcement * grad_logits
            total_weight_step += np.outer(obs, step)
            total_bias_step += step
        self.action_weights += total_weight_step / n
        self.action_bias += total_bias_step / n
        self._pending = []
        self._n_zero = 0

    def clear_pending(self) -> None:
        """Discard any buffered-but-not-yet-applied examples -- call before
        reusing a policy object across a driver reset (a Codex review caught
        that a mid-batch reset, e.g. after an exception between accumulate()
        and apply_batch(), would otherwise let stale transitions from the old
        run leak into the new one's first update)."""
        self._pending = []
        self._n_zero = 0

    def action_weight_absmean(self) -> float:
        return float(np.abs(self.action_weights).mean())
