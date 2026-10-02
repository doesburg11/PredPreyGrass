# Linear SARSA(lambda) for within-lifetime Baldwin learning

## Is SARSA(lambda) genuinely distinct and theoretically suitable?

### Takeaway
Yes. Linear SARSA(lambda) is a materially different scientific treatment from the present learner: it learns an action-value function by bootstrapped temporal-difference errors and derives behavior from those values, whereas the current code directly changes a softmax policy in the direction of the immediately reinforced action. It preserves the experiment's key Baldwin separation if the inherited parameters initialize a private lifetime action-value estimator and only the inherited initialization is reproduced.

### Cited Findings
- SARSA(lambda) is the on-policy control counterpart of TD(lambda): it estimates action values rather than state values, and its behavior policy depends on those evolving action-value estimates. — [van Seijen et al., *True Online Temporal-Difference Learning*](https://arxiv.org/abs/1512.04087)
- The original “modified connectionist Q-learning” work was explicitly motivated by extending reinforcement learning from finite discrete states to high-dimensional continuous state spaces with function approximation. — [Rummery & Niranjan, *On-line Q-learning using connectionist systems*](https://citeseerx.ist.psu.edu/document?doi=7a09464f26e18a25a948baaa736270bfb84b5e12&repid=rep1&type=pdf)
- TD with linear function approximation learns from an actual online trajectory and, for fixed-policy prediction under its assumptions, has almost-sure convergence and a characterized approximation limit. Those results do **not** directly prove convergence of changing-policy SARSA control. — [Tsitsiklis & Van Roy, *An Analysis of Temporal-Difference Learning with Function Approximation*](https://web.mit.edu/~jnt/www/Papers/J063-97-bvr-td.pdf)
- Modern theory does not justify a blanket convergence claim for ordinary linear SARSA control: projected SARSA may “chatter” in a bounded region, and available guarantees require restrictive conditions. — [Zhang, Tachet des Combes & Laroche, *On the Convergence of SARSA with Linear Function Approximation*](https://arxiv.org/abs/2202.06828)

### Inferences
- The current repository's `reinforce_update` is closer to a one-step, immediate-reinforcement policy-gradient update than canonical episodic REINFORCE: it applies `reinforcement * grad log pi(a|s)` directly, with no return-to-go or learned critic. SARSA(lambda) therefore changes both the learned object (Q rather than policy logits) and temporal credit mechanism (bootstrapped TD plus traces), making it distinct enough for a meaningful comparison. — [local implementation: `networks.py`](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/networks.py)
- The fairest Baldwin mapping is: genome `action_weights/action_bias` = inherited initial Q parameters; agent live copies = non-inherited lifetime Q parameters; eligibility traces = zeroed at birth and never inherited. Reproduction already reads `agent.genome`, not the live copies, so this separation fits the existing architecture. — [local architecture: `README.md`](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/README.md)
- This changes the behavioral interpretation of inherited action weights from “policy logits” to “initial action values.” That is acceptable for an alternative-algorithm treatment, but results should not be described as differing *only* in optimizer.

### Gaps
- There is no theorem establishing convergence for this project's setting: observations are aliased/partial, agents and ecology change concurrently, lifetimes are short random horizons, and the policy changes online.
- Scientific distinctness is a design judgment rather than a fact proved by the literature; it should be stated operationally in the preregistration (direct policy update versus value-based TD control).

## What exact online formulation should be used?

### Takeaway
Use linear, action-block semi-gradient SARSA(lambda) with accumulating traces as the minimal first implementation, a persistent epsilon-soft policy, and explicit terminal updates. Treat `gamma`, `lambda`, `alpha`, and `epsilon` as algorithm parameters fixed across conditions or tuned on separate seeds. A softmax-over-Q policy is a defensible alternative if preserving smooth stochastic action selection matters more than canonical comparability.

### Cited Findings
- In SARSA(lambda), eligibility is attached to state-action pairs; the accumulating trace is `e_t = gamma*lambda*e_{t-1} + phi(S_t,A_t)`, and action-value learning conditions the expected return on both state and action. — [van Seijen et al., *True Online Temporal-Difference Learning*](https://arxiv.org/abs/1512.04087)
- True-online SARSA(lambda) is the exact online-forward-view variant; experiments in its primary paper found it at least as good as conventional accumulating/replacing variants in the tested cases, while all variants coincide at `lambda=0`. — [van Seijen et al., *True Online Temporal-Difference Learning*](https://arxiv.org/abs/1512.04087)
- A recent random-horizon convergence result requires an epsilon-soft policy that is Lipschitz in the weights with a sufficiently small Lipschitz constant; its analyzed variant updates weights/policy only at trajectory ends, so it is not a theorem for the proposed per-step implementation. — [Palmborg, *Convergence of SARSA with linear function approximation: The random horizon case*](https://arxiv.org/abs/2306.04548)

### Inferences
- Define `x_t = concat(obs_t, 1)` and `Q_t(a) = x_t @ theta[:,a]`, where `theta` has shape `(obs_dim+1, 4)` (or retain separate `(W,b)`). The equivalent action-block feature `phi(s,a)` is zero in every action block except block `a`, which equals `x`. For each nonterminal transition:
  - select `A_{t+1}` from the **same current behavior policy** used to act;
  - `delta_t = r_{t+1} + gamma*Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)`;
  - `e_t = gamma*lambda*e_{t-1} + phi(S_t,A_t)`;
  - `theta <- theta + alpha*delta_t*e_t`.
- Canonical exploration: choose uniformly among all four actions with probability `epsilon`, otherwise choose uniformly among maximizing actions. Keep a nonzero epsilon during life; rapidly annealing it changes the learned on-policy target and can eliminate continued exploration. To stay closer to the current stochastic policy, use `pi(a|s) = softmax(Q(s,a)/tau)` with fixed `tau`; this is still on-policy provided the sampled `A_{t+1}` is the action used in the target and subsequently executed.
- Ordering matters in `world.py`: observe `S_{t+1}`, compute its evaluation/reward, choose and cache `A_{t+1}`, update the previous transition using that exact action, then execute `A_{t+1}`. Selecting a fresh action after the update would make the target action differ from the behavior action and cease to be literal SARSA. — [local step loop: `world.py`](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py)
- With the existing `r_{t+1}=E(S_{t+1})-E(S_t)`, `gamma=1` makes an undiscounted finite-life return telescope to `E(S_T)-E(S_0)`. With `gamma<1`, intermediate evaluation values no longer cancel completely, so gamma is part of the proximate objective, not merely a numerical setting. Because traces decay by `gamma*lambda`, use a finite-horizon interpretation and test both a near-undiscounted setting and a moderate discount rather than silently adopting `0.99`.
- Terminal transition: when death is attributable to an executed action/environment step, perform `delta = r_terminal - Q(S_t,A_t)` with bootstrap zero, update the trace/weights, then clear the trace. The repository currently has no well-defined post-death observation/evaluation and dead agents do not reach the next loop, so the experiment must define `r_terminal` explicitly (for example, a fixed inherited-evaluation-independent death penalty, or an evaluation of a documented absorbing vector). Omitting it reproduces the present learner's blind spot but prevents SARSA from learning directly from lethal actions.
- Start with accumulating traces because it is the transparent textbook backward view and continuous features make “replacing” traces less natural. Add true-online SARSA(lambda) only if large alpha/lambda sensitivity appears; doing both initially adds an unnecessary algorithm factor.
- Normalize/clip features and optionally clip TD error or parameter norm as diagnostic safety measures, but preregister them: clipping changes the algorithm. The current observations are mostly bounded `[0,1]`, which is favorable, but Q values and accumulating traces are not inherently bounded.

### Gaps
- The correct terminal reward is not implied by `E_t-E_{t-1}` because the code does not construct a terminal observation. This must be a scientific choice, not an implementation accident.
- Whether epsilon-greedy or softmax-over-Q is the fairest behavior-policy match is unresolved. Epsilon-greedy is canonical and easier to explain; softmax preserves the existing action-sampling family and is smoother in the parameters.
- Separate positive/negative learning rates in the current learner have no standard SARSA counterpart. A single alpha is cleaner; retaining asymmetric alphas would introduce another nonstandard treatment.

## How does it fit the repository, and what are the main risks and tests?

### Takeaway
The storage footprint is tiny and the inherited/live separation already exists, but correct SARSA integration touches `Agent` state, birth/reproduction initialization, and action/update ordering. The largest scientific risks are reward telescoping/terminal ambiguity, partial observability, nonstationarity from other agents, extrapolation from shared linear features, and confounding algorithm choice with exploration.

### Cited Findings
- Online sampling along the actual Markov-chain trajectory is important for the positive linear-TD result; sampling from an unrelated distribution can diverge. This supports direct online updates and argues against adding replay. — [Tsitsiklis & Van Roy, *An Analysis of Temporal-Difference Learning with Function Approximation*](https://web.mit.edu/~jnt/www/Papers/J063-97-bvr-td.pdf)
- Concurrent learners make each agent's experienced environment effectively nonstationary because other agents' policies also change. — [Kim et al., *A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning*](https://proceedings.mlr.press/v139/kim21g.html)
- Even linear on-policy SARSA control has limited general guarantees; projection can bound iterates, but convergence may only be to a bounded region under broad conditions. — [Zhang, Tachet des Combes & Laroche, *On the Convergence of SARSA with Linear Function Approximation*](https://arxiv.org/abs/2202.06828)

### Inferences
- Minimal repository changes for a later implementation:
  - `networks.py`: add Q computation, epsilon-soft/softmax-Q action selection, and SARSA(lambda) update helpers;
  - `Agent`: add an eligibility matrix/bias trace (or combined `(obs_dim+1,4)` trace) initialized to zero;
  - founder/offspring construction: copy genomic action parameters into live Q parameters and zero traces;
  - `_step_agents`: choose next action before updating the previous transition; execute that cached action;
  - death/removal paths: issue a terminal update exactly once and clear trace;
  - checkpointing: confirm the new live trace/state is either serialized for exact continuation or intentionally reconstructed, while genomes remain unchanged;
  - config/logging: add `sarsa_alpha`, `gamma`, `lambda`, exploration parameter, TD-error/Q/trace diagnostics, and algorithm identifier. — [local agent and loop: `world.py`](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py); [local configuration: `config.py`](/home/doesburg/Projects/PredPreyGrass/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/config.py)
- Existing `action_site_change_rate` remains interpretable as constraint on inherited initial Q parameters, not direct policy parameters. The write-up and plots must relabel that interpretation. Lifetime-learning diagnostics should separately log mean absolute TD error, Q scale, trace norm, update norm, action entropy, and learned-minus-genomic parameter norm.
- Required unit tests: exact hand-calculated one-step update; trace propagation across two transitions; only selected-action feature block enters the new trace; target uses the actually cached next action; terminal bootstrap is zero and occurs once; birth zeros traces; reproduction never inherits live Q or traces; learning-disabled conditions do not mutate live parameters; fixed-seed tie handling/exploration is reproducible; checkpoint resume is trajectory-identical if exact continuation is promised.
- Experimental controls should include existing learner, SARSA(lambda), evolution-only, learning-only, and neither; additionally include SARSA(0) to isolate eligibility-trace credit assignment. Match action stochasticity/entropy as closely as possible or report it, because epsilon versus the current softmax is otherwise a major confound.
- Stability protocol: first run deterministic micro-MDP tests; then a frozen-world/single-agent diagnostic; then short ecology sweeps over log-spaced alpha, a small lambda set, gamma, and exploration; reject settings with exploding Q/trace norms; finally evaluate multiple held-out seeds. Do not infer theoretical safety merely because the approximator is linear.
- Partial observability means identical seven-dimensional observations can require different actions based on hidden layout/history. A linear memoryless Q function cannot resolve that aliasing; lambda improves temporal credit but does not add memory. This is a representational ceiling shared in part with the current linear policy.
- Nonstationarity is unusually strong here: ecological state changes, other agents reproduce/die/learn, and the observation omits identities. SARSA's on-policy nature avoids the classic off-policy mismatch but does not make the environment stationary or Markov.

### Gaps
- The repository does not currently expose a discrete “episode end” callback per agent, so every death path must be audited before implementation to prevent missing or duplicate terminal updates.
- Appropriate hyperparameter ranges require empirical scale measurements of `E_t-E_{t-1}`, feature norms, and lifespan distributions; primary literature cannot determine them for this ecology.
- The existing checkpoint implementation was not exhaustively traced in this research pass, so exact live-state serialization requirements need a separate implementation audit.
