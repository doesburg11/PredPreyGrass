# Design note: nature, nurture and both in the deer stag hunt (2026-11-08)

Status: **draft 2, awaiting the user's decision; nothing built.** Draft 1 was
revised after a Codex review (see "Revisions after review" at the end). The open
decisions are listed in the section before it.

## Why

deer_stag_hunt replicated (RESULTS.md section 1.1). Hunters and deer coexist,
and hunters approach adult deer with a partner near more often than chance
(1.15-1.23 times). That result comes from learning alone: one shared PPO policy
per species, with no genomes.

The site's tagline is "the nature and nurture of cooperation". The project's
ideal is cooperation that comes from heritable traits and learning together,
doing better than either alone. Within PredPreyGrass that has been shown only
in the discrete ERL modules (Trials 11 and 14), and not for cooperation.
continuous_traits showed that learning steers body evolution, but it had no
three-way comparison and no cooperation.

This stage asks one narrow question:

> In the deer world, against fixed deer, do hunters with heritable innate
> attractions plus shared learning catch more adult deer, and more per attempt,
> than hunters with shared learning only? And than hunters with the innate
> attractions only?

## Proposal

### A new module

The new module is `evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt`,
a copy of deer_stag_hunt's world with three changes:

1. a genome per hunter, with inheritance and a carried-over founder pool;
2. the hunter's own genes added to its observation;
3. an innate steering bias added to its action logits.

Catching, energy, births and the rest of the observation are unchanged. With
genes not expressed and the gene inputs removed, an episode is identical to one
in deer_stag_hunt for the same actions (tested, as in test_equivalence.py).

### Genes: neutral innate attractions

Each hunter carries four genes, each bounded to [-1, 1]:

| Gene | Effect |
| --- | --- |
| `a_hunter` | attraction (+) or avoidance (-) toward hunters in view |
| `a_young` | the same toward fawns and middle deer in view (energy below 5) |
| `a_adult` | the same toward adult deer in view (energy 5 or more) |
| `neutral` | none |

- **Neutral start:** the first founders draw every gene from N(0, 0.1). Nothing
  is built in toward hunting together.
- **Inheritance:** a newborn inherits its parent's genes plus N(0, 0.05) per gene,
  clipped. These are continuous_traits' values. Reproduction stays asexual, as in
  deer_stag_hunt.
- **Founders:** as in continuous_traits (`_update_gene_pool`), each simulation
  environment keeps its own lineage. The founders of its next episode are the
  hunters alive at the end of this one. If hunters died out, they are recent
  births; if there were none, the previous pool.
  - Genomes are not exchanged between environments during training. Each run is
    a set of separate populations, all trained by one shared policy.
  - Genomes are pooled across environments only when a checkpoint is saved,
    which is what evaluation uses.
  - Selection therefore acts through births and through survival to an
    episode's end.
- **No energy cost:** no gene changes energy, speed or sight; the genes act on
  behaviour only.
- **Only hunters carry genes:** cooperation is the hunters' question.

### How the genes act

A hunter's observation already holds the 4 nearest hunters and 2 nearest deer in
view. Each slot has a seen flag and an offset scaled by view distance, and each
deer's energy is scaled by 16. Using only slots whose seen flag is 1, the innate
desire is

    d = a_hunter · mean(unit offsets of seen hunters)
      + a_young  · mean(unit offsets of seen deer with scaled energy < 5/16)
      + a_adult  · mean(unit offsets of seen deer with scaled energy ≥ 5/16)

A kind that is not in view adds nothing. Each of the 48 moves has a direction
`u_m` and a speed `s_m`: 16 directions at full speed, 16 at half, and 16 stop
moves (all identical). Its innate bias is

    bias_m = β · s_m · (u_m · d)

so moves along `d` are favoured, moves against it disfavoured, and stop moves
get no bias. The hunter samples its move from

    logits_m = λ · net_m(obs, genes) + bias_m

### Three arms

| Arm | λ | β | Genes in observation | What decides a move |
| --- | --- | --- | --- | --- |
| **E**, evolution only | 0 | β₀ | (no network) | genes only, plus a fixed cruise rule below |
| **L**, learning only | 1 | 0 | yes, unexpressed | the PPO policy |
| **B**, both | 1 | β₀ | yes, expressed | the PPO policy plus the hunter's own bias |

- **Genes visible in L and B.** The network and the value function see the
  hunter's own four genes in both arms, so L and B have identical networks and
  inputs. The difference is only whether the genes act. Hidden genes would make
  B harder to learn than L: the shared policy could not tell which hunter's bias
  it has to build on or offset. In L the genes are inherited and mutated but do
  nothing: they are the arm's drift reference.
- **PPO stays on-policy.** The bias is computed inside the predators' RLModule
  from the stored observation, so the same `net + bias` logits produce actions,
  stored log-probabilities, PPO ratios, entropy and KL. Tests check this.
- **E's fixed cruise rule:** with all logits at zero, a hunter would stop with
  probability 1/3, because 16 of the 48 moves are stops. E therefore uses a fixed
  rule:
  - when no hunter or deer is in view, it moves at full speed along its current
    heading;
  - otherwise it samples from `bias` with the stop moves masked out.

  This gives genes alone a fair chance of finding deer. The rule is the same in
  every E run, and it isn't heritable.
- **What E cannot do.** E hunters can't use energy, cooldowns, grass or pursuit
  prediction. A B > E gap therefore shows only that these four genes alone are
  not enough here. The informative contrast is B against L.

### β fixed in advance, not tuned on results

β₀ is fixed before any run by one behavioural rule, computed on a synthetic
observation:

- **the case:** a hunter with one gene at +1, one target of that kind in view,
  network logits all zero;
- **the rule:** the probability of a full-speed move within 22.5° of the target's
  direction is 0.5.

β₀ follows from the 48-move geometry alone: β₀ ≈ 3.44, against a probability
of 1/16 without the bias. The pilot never changes it.

### Deer: frozen, one per deer_stag_hunt run

The deer are frozen in every arm. If they learned in each arm, a difference
between arms could come from the deer adapting differently to E, L and B hunters.

- **Which policy:** run k of each arm uses the final deer policy of deer_stag_hunt
  run ((k − 1) mod 3) + 1. So arms are matched on deer, and no single,
  favourable deer policy is picked after the fact.
- **Sampling:** deer act stochastically, as in training (the policy is an MLP,
  with no hidden state).

This stage is about nature versus nurture in hunters, not coevolution.

## Experiment

### Runs and budget

- **Runs:** 5 runs per arm, 15 in all.
- **Seeds:** seed k is used for run k in every arm; it is also the deer pairing.
- **L and B:** 1000 training iterations, hunters starting from a random network.
- **E:** E needs no learner. It runs as plain parallel episodes on the CPU, with
  the same number of simulation environments. Each E run is matched to the median
  total physics steps of the L and B runs at iterations 500, 750 and 1000.
- **Recorded per arm:** physics steps, completed episodes, births and generations,
  so that evolutionary opportunity can be compared directly.

### Measures (evaluate.py, 50 episodes per checkpoint)

- **Co-primary measures:**
  - **A, adults caught per episode:** doing better, through the bigger food
    source.
  - **S, adult attempt success:** adults caught ÷ (adults caught + failed
    attempts). A failed attempt is a ready hunter touching an adult without a
    second toucher within the window; it starts a cooldown (deer_stag_hunt
    `callbacks.py`). This is coordination per attempt, less tied to how many
    hunters are alive.
- **Supporting measure:** C, adult approaches with a partner near ÷ chance, from
  coop_fair.py, 20 episodes.
- **Run value:** each measure's mean over checkpoints 500, 750 and 1000. These
  three are one run's repeated measures, not independent data.
  - **Estimand:** sustained late-training performance, not final performance.
  - **Why:** deer_stag_hunt's late decline (RESULTS.md section 1.1) showed that a
    single final checkpoint can be decided by where a decline happens to fall.
    This is a new experiment, so the choice is made before its data.
- **Unconditional:** every episode counts. After hunters die out, the episode
  simply adds no more catches. An episode with no attempt on an adult adds
  nothing to S's numerator or denominator; S is pooled per run as total catches ÷
  total attempts.
- **Evaluation:** founders are drawn from the gene pool saved with the
  checkpoint, mutation is on, and actions are sampled as in training.

### Tests, set before the main runs

All tests are one-sided Mann-Whitney tests on run values, 5 against 5, at
p < 0.05. That allows up to 4 of the 25 run pairs to be out of order (p = 0.048).

1. **B beats L** on A and on S. Both must hold.
2. **B beats E** on A and on S. Both must hold.
3. **B sustains coexistence:** in at least 4 of 5 B runs, hunters and deer are
   alive at the end of at least 45 of 50 episodes, averaged over the three
   checkpoints. The point is that better hunting doesn't come from eating the
   deer out.

**Verdict:** if tests 1-3 are all met, the result is: "in this world, against
fixed deer, an innate attraction bias that evolves plus shared PPO learning
catches more adults, and more per attempt, than learning alone or the bias
alone."

**Also reported:**

- each run's values, with the difference in arm means and a bootstrap 95%
  interval over runs;
- C for every arm, and whether B's C exceeds 1 in all its runs;
- coexistence for every arm.

### Also recorded, not tests

- **Selection on genes:**
  - per generation, the selection differential cov(relative offspring, gene),
    using continuous_traits' `selection_differentials`;
  - per run and gene, the mean of these over the run, with the same quantity
    for L's unexpressed genes as the drift reference.
  - The `neutral` gene is reported too. With asexual inheritance it hitchhikes
    with selected genes, so it is not a clean drift control on its own; L's whole
    unexpressed genome is the reference.
- **Learning steering genes:** whether gene means differ between B and E. This
  takes a between-arm comparison of run values, not a look at trajectories.
- **Other measures:** hunter births, the adults' share of hunters' energy, and
  checkpoints 250 and 500 on their own.

### What outcomes would mean, written in advance

- **Tests 1-3 met:** the verdict above, and no more. It speaks for this one bias
  mechanism, these frozen deer and this measure, not for nature and nurture in
  general.
- **B ≈ L, with B's selection differentials near L's:** learning hides the genes
  from selection, because it does the job whatever the genes are. That is a known
  effect, the opposite of a Baldwin effect, and a useful negative.
- **B < L:** the innate bias gets in the way of learning.
- **E fails (hunters extinct in most episodes):** genes alone in this form aren't
  enough. It doesn't block tests 1 and 3.

## Pilot and hard stop

Following [[avoid endless tweaking]], the limits are agreed before anything runs.

- **Pilot:** 1 run per arm, 500 iterations. It checks only that:
  - nothing crashes;
  - the stored log-probabilities match `net + bias`;
  - L and B don't collapse (hunters alive at the end in most evaluated episodes).
- **No tuning:** β and every other setting stay fixed. If the pilot fails, the
  stage stops; nothing is adjusted and re-run.
- **Pre-registration:** the main runs are written into the new module's
  RESULTS.md section 1 before they start.
- **Stop:** after the main runs, the experiment stops whatever the outcome.
  Follow-ups are new decisions: deer evolving and learning again, other traits,
  social learning.

## Cost

- **One run:** about 5-6 h alone in deer_stag_hunt, and about 7 h each when two
  share the GPU. With frozen deer, only one policy learns, so runs may be faster.
- **Pilot:** about one day.
- **Main runs:** 10 GPU runs, 2 at a time, about 35 h; the E runs are CPU-only
  and shorter.

## Limits, set in advance

- **Not a classic Baldwin effect:** learning is one PPO policy shared by all
  hunters, so it is species-level learning, not each hunter learning in its
  lifetime. The genes, though, are individual. A positive result means
  "individual innate variation plus shared learning". Write-ups must not call
  the shared policy culture or individual learning.
- **No coevolution:** the deer are frozen.
- **One mechanism:** an additive attraction bias with 4 genes. Other traits
  (sharing tolerance, social attention) or another way for genes to act could
  give other results.
- **Not partner seeking:** C, adult approaches with a partner near, is a measure
  of company, not proof that hunters seek a partner (deer_stag_hunt RESULTS.md
  section 1.1).
- **Statistics:** five runs per arm show run-to-run variation. They give a test,
  not a population estimate.
- **Social learning is a later stage:** payoff-biased copying of these same genes
  (dual inheritance) would add a fourth way to change a gene.

## Implementation sketch (after approval)

- **Environment:** deer_stag_hunt's world plus genomes (`family`, gene pool,
  archive, selection differentials) adapted from continuous_traits. Four gene
  inputs are added to the predators' observation.
- **RL module:** a torch RLModule for predators that computes
  `λ · net(obs) + β · bias(obs)` inside its forward passes, for both exploration
  and training. The value function sees the same inputs.
- **E runner:** an episode runner without RLlib, using the same environment and
  the cruise rule.
- **Tests:**
  - with genes unexpressed and gene inputs dropped, episodes match deer_stag_hunt
    for the same actions;
  - the bias uses seen flags only and the 5/16 energy threshold, and each gene's
    bias points the right way, by sign and kind;
  - β₀ meets its 0.5 rule;
  - stored log-probabilities equal a categorical distribution from `net + bias`,
    and the PPO ratio is 1 before an update;
  - offspring inherit genes with mutation, and founders come from survivors, as
    in continuous_traits;
  - E's cruise rule and its masked stop moves.
- **Review:** Codex reviews the code before any run.

## Decisions for the user

1. **The design as a whole:** a new module, three arms, neutral attraction genes
   acting as an additive bias, genes visible to the network.
2. **Deer frozen, matched by run** (recommended), or deer learning in all arms,
   which brings back coevolution but makes arm differences harder to attribute.
3. **Young deer as one kind** (recommended: only two deer slots), or fawns and
   middle deer as separate genes.
4. **5 runs per arm** (recommended, about 35 GPU hours), or 4 (then every B run
   must beat every L run, p = 1/70, and one near tie reverses the verdict).
5. **E's cruise rule** (recommended), or a pure gene-only E that stops a third of
   the time when nothing is in view.

## Revisions after review (Codex, 2026-11-08)

Draft 1 was reviewed by Codex; the changes:

- **Genes visible to network and value function in L and B.** Draft 1 hid them,
  which would have made B harder to learn than L.
- **Stop moves:** E's random fallback stopped a third of the time; it is replaced
  by a cruise rule with masked stops.
- **β fixed by a behavioural rule,** no longer chosen in the pilot by whether
  genes moved.
- **Primary measure:** adults caught per episode is joined by adult attempt
  success, since catch counts alone grow with hunter numbers.
- **Guard:** draft 1 had an asymmetric coexistence guard. It is now unconditional
  measures for every arm, plus a B coexistence test.
- **Gene pool:** the description was corrected to per-environment lineages from
  survivors, with no exchange during training.
- **Drift reference:** the neutral locus hitchhikes; selection is now judged by
  selection differentials against L's unexpressed genes.
- **Deer:** run_2's deer were picked after the fact; deer are now matched by run
  across arms.
- **Statistics:** 4 runs with all-above-all became 5 runs with Mann-Whitney, which
  tolerates near ties. Seeds are shared across arms, and a bootstrap interval is
  reported.
- **Cooperation test:** draft 1's misstated ratio interval is now a supporting
  measure C, reported per run.
- **Verdict wording** narrowed to this mechanism, these deer and these measures.
- **Budget** matched on physics steps, with episodes, births and generations
  recorded.
