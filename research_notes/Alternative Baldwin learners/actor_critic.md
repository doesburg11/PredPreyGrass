# Minimal linear actor–critic for the ERL/Baldwin model

## Which minimal actor–critic formulation is appropriate?

### Takeaway
Use an on-policy, one-step continuing actor–critic with a linear state-value critic and the existing softmax linear actor. Treat the critic as lifetime-only state initialized to zero at birth; keep the actor’s inherited initial parameters separate from its lifetime-updated copy. Start with TD(0), not traces or average reward: the existing difference reward makes a conventional average-reward objective degenerate in a stationary continuing regime, while traces add a credit-horizon parameter before the one-step mechanism is validated.

### Cited Findings
- The policy-gradient theorem supports an explicitly parameterized stochastic policy updated using an approximate action-value or advantage function; the original paper identifies both REINFORCE and actor–critic as members of this same policy-gradient family. — [Sutton et al., *Policy Gradient Methods for Reinforcement Learning with Function Approximation*](https://papers.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- Konda and Tsitsiklis formulate actor–critic as a two-time-scale method: a linear TD critic supplies an approximate gradient direction to the actor, and the critic’s features must be related appropriately to the actor parameterization. — [Konda & Tsitsiklis, *Actor-Critic Algorithms*](https://papers.nips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- TD(λ) uses eligibility traces to assign a later TD error backward across earlier states; λ interpolates the temporal-credit horizon. — [Sutton, *Learning to Predict by the Methods of Temporal Differences*](https://link.springer.com/content/pdf/10.1007/BF00115009.pdf)
- The current repository already uses a linear softmax actor and updates the live, non-genomic weights with `reinforcement * grad log pi`; its reinforcement is `E_t-E_{t-1}`. — [repository `networks.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/networks.py#L1-L73)
- At birth the live actor is copied from genomic action weights; reproduction constructs a child from genomic rather than lifetime-learned weights. — [repository `world.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py#L398-L411); [reproduction path](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py#L803-L854)
- Average-reward RL is intended for continuing control and models response vigor using reward rate, but that objective is ill-suited to a pure temporal-difference reward here. — [Niv et al., *Tonic dopamine: opportunity costs and the control of response vigor*](https://www.princeton.edu/~ndaw/ndjd07.pdf)

### Inferences
- Minimal equations at transition `(x_t,a_t,r_{t+1},x_{t+1})` are: `V_w(x)=w·x+b`; `δ_t=r_{t+1}+γV_w(x_{t+1})-V_w(x_t)`; critic `w←w+βδ_t x_t`, `b←b+βδ_t`; actor `θ←θ+αδ_t∇θ log πθ(a_t|x_t)`. The existing four-action softmax supplies the last gradient exactly.
- With repository timing, `r_t=E(x_t)-E(x_{t-1})`; therefore the update already has `(prev_obs, prev_action, current_obs)` when it runs. No transition buffer or environment API redesign is needed.
- A critic vector of `obs_dim` plus one bias is enough. It should be zeroed for every founder/newborn and never added to `Genome`; actor live weights remain copied from the inherited genome exactly as now. This is a clean Baldwin separation: inherited prior policy, acquired actor and critic state, no transmission of acquired parameters.
- Do **not** use differential/average-reward actor–critic initially. In a stationary trajectory, the long-run mean of `E_t-E_{t-1}` telescopes to zero (bounded endpoint difference divided by horizon). Its learned average reward would therefore converge toward zero and carry little task information.
- Discounting also needs explicit interpretation. For `0<γ<1`, the return from difference rewards is not simply future goodness: `Σ γ^k(E_{t+k+1}-E_{t+k}) = -E_t + (1-γ)Σ γ^{j-1}E_{t+j}`. Thus γ changes the scientific objective. A small sweep such as `{0, .5, .9, .99}` is more honest than silently adopting `.99`; `γ=0` tests whether the critic baseline alone helps, while larger γ tests predictive temporal credit.
- If TD(0) shows a signal, add separate actor and critic traces: `e^V←γλ_V e^V+x_t`; `e^π←γλ_π e^π+∇logπ`; update with `δ`. Use replacing/dutch-style bounded variants only if accumulating traces visibly explode. λ should not be the first source of complexity.

### Gaps
- No primary source validates actor–critic specifically with a genetically evolved potential-like signal `E_t-E_{t-1}` in an ecological birth/death simulation; the objective implications above are algebraic deductions.
- Death has no explicit terminal update in the current loop. A scientific choice is required: treat death as terminal with `V=0` and update the fatal transition, or preserve current semantics, which may omit the strongest negative outcome.

## Does a critic provide a substantively different test, and is it biologically/evolutionarily defensible?

### Takeaway
It is a meaningful mechanistic control but not a conceptually independent alternative to REINFORCE. The actor update remains the same score-function policy gradient; the novelty is a learned TD prediction error that bootstraps and acts as a state-dependent baseline. Its biological case is stronger than an arbitrary optimizer, but still a computational analogy rather than evidence that this simulated ecology implements neural anatomy.

### Cited Findings
- Sutton et al. explicitly place REINFORCE and actor–critic in the same policy-gradient class; actor–critic uses approximate value/advantage information to estimate the gradient. — [Sutton et al.](https://papers.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
- Konda and Tsitsiklis characterize actor-only methods as having high-variance gradient estimates and actor–critic as using a learned critic to obtain faster/less variable improvement directions, subject to approximation and two-time-scale conditions. — [Konda & Tsitsiklis](https://papers.nips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- Actor–critic models of basal ganglia are motivated by the resemblance between dopaminergic activity and a critic’s TD prediction error, and between dopamine-dependent striatal plasticity and actor learning. — [Joel, Niv & Ruppin, *Actor–critic models of the basal ganglia*](https://doi.org/10.1016/S0893-6080(02)00047-3)
- Physiological work summarized in the three-factor literature supports synaptic eligibility traces plus a later neuromodulatory factor over behavioral timescales, although it does not uniquely establish actor–critic. — [Gerstner et al., *Eligibility Traces and Plasticity on Behavioral Time Scales*](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00053/full)

### Inferences
- Relative to the repository update `θ←θ+α r_t ∇logπ`, one-step actor–critic changes only the scalar modulator to `δ_t=r_t+γV(x_{t+1})-V(x_t)`. It directly tests whether predictive credit/baseline subtraction fixes learning, but it does **not** test a different policy representation or a non-policy-gradient learning principle.
- It is more substantively different when `γ>0`, because the critic bootstraps expected future changes in innate evaluation. With `γ=0`, it is principally a learned state baseline and is closest to variance-reduced REINFORCE.
- Evolutionary defensibility is clean: evolution specifies the innate evaluator and initial actor; lifetime experience learns both an outcome predictor (critic) and action preferences; only genomic evaluator/initial actor parameters reproduce. This creates no Lamarckian channel.
- Biological defensibility is moderate: actor/critic separation, TD error, and three-factor modulation have established neuroscientific analogies. The exact linear features, globally shared scalar δ, learning rates, and reset-to-zero critic are engineering abstractions, so claims should say “biologically inspired,” not biologically realistic.
- If the research question demands a truly independent algorithmic alternative to the current REINFORCE-style rule, linear SARSA(λ) or another value-control method is the stronger contrast. Actor–critic is best framed as a diagnostic bridge: does learned prediction/temporal credit rescue the same actor gradient?

### Gaps
- The sources support actor–critic as a model class and neural analogy, not the specific inheritance scheme proposed here.
- There is no principled biologically grounded value for `α`, `β`, `γ`, or `λ` for this simulation; these remain experimental hyperparameters.

## What stability, credit-assignment, and repository-integration risks arise?

### Takeaway
The main risks are critic bias from partial observability, a moving multi-agent ecology, poorly separated actor/critic learning timescales, and semantic distortion caused by discounting a difference reward. Integration is small in code footprint but must include terminal transitions, lifetime resets, metrics, and controlled ablations to make results interpretable.

### Cited Findings
- Konda and Tsitsiklis’ convergence analysis relies on two-time-scale learning, with the critic operating on the faster scale and actor steps becoming comparatively slower; it also imposes feature/regularity assumptions. — [Konda & Tsitsiklis](https://papers.nips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- Even linear function approximation can become unstable for some RL/control combinations; Baird’s classic result warns that tabular guarantees do not automatically transfer to function approximation. — [Baird, *Residual Algorithms: Reinforcement Learning with Function Approximation*](https://leemon.com/papers/1995b.pdf)
- Standard MDP theory assumes a stationary environment, whereas interacting adaptive agents motivate a Markov-game formulation. — [Littman, *Markov Games as a Framework for Multi-Agent Reinforcement Learning*](https://www.eecs.harvard.edu/cs286r/courses/spring06/papers/littman94markov.pdf)
- The repository observation has only directional occupancy-like visual signals plus tree, health, and energy features, while action consequences also depend on hidden object identity and population activity; thus identical observation vectors can precede different outcomes. — [repository observation/action code](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py#L431-L565)
- The current agent already stores `prev_obs`, `prev_action`, and `prev_eval`, and the action update occurs after observing the next state but before sampling the next action. — [repository `world.py`](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py#L210-L223); [step path](https://github.com/doesburg11/PredPreyGrass/blob/main/predpreygrass/evolutionary/eco_evolutionary_erl_baldwin/world.py#L505-L531)

### Inferences
- Integration plan: add lifetime-only `critic_weights`, `critic_bias` (and later optional traces) to `Agent`; initialize/reset them for founders and newborns; compute δ at the current update site; update critic from `prev_obs`; update the existing live actor from `prev_obs/prev_action`; never flatten, mutate, cross over, checkpoint as genome, or inherit critic state.
- Preserve asymmetric positive/negative actor rates only as a planned ablation. The clean canonical actor–critic has one actor step size; retaining `lr_positive/lr_negative` improves comparability with current CRBP-inspired behavior but makes it less canonical and expands tuning.
- Suggested conservative first grid: actor α `{0.002, .01, .05}`, critic β `{2α, 5α, 10α}`, γ `{0, .5, .9, .99}`, with gradient/δ logging and optional clipping as a safety diagnostic. Do not claim two-time-scale theory from these finite constant rates; use it only as design guidance.
- Instrument per-agent and aggregate `|δ|`, critic `V` mean/std/max, actor drift from genome, critic parameter norm, policy entropy, and correlations among `r`, `δ`, and subsequent evaluation changes. These reveal critic explosion, collapse, or a baseline that simply cancels the already sparse signal.
- Required ablations: current direct update; actor–critic with γ=0; actor–critic with best γ>0; frozen/random critic control; evolution-only; and actor–critic with critic zeroed at evaluation. Use matched seeds and report extinction/survival, reproduction, evasion, and genetic-assimilation metrics.
- Partial observation aliases different latent situations, so a linear `V(obs)` can produce systematically biased δ. Traces may improve delayed credit but cannot reconstruct hidden state; adding recurrence would be a materially larger architecture change.
- Concurrently learning agents, evolving genomes, births/deaths, plants, and carnivores make the transition distribution nonstationary. A fast critic may chase that distribution; a slow critic may never become useful within short lifetimes. Analyze learning curves by age, not just global step.
- Since agents may die during action resolution before the next normal update, explicitly route a terminal TD update through the death path. Otherwise lethal actions receive only whatever immediate observation change occurred before death, reproducing the present credit hole.

### Gaps
- Exact typical lifetime lengths and the empirical distribution/autocorrelation of `E_t-E_{t-1}` were not measured in this read-only research task; those determine whether a critic can converge before death.
- The GitHub links reflect the inspected `main` paths; if the remote repository is private, report readers need local checkout access.
