# Cooperation Rate as a Genome Trait

## The mechanism

`cooperation_rate` is a heritable trait, bounded `[0.0, 1.0]`. Each step, when an agent
successfully hunts (predator) or grazes (prey), it donates a fraction of *that catch* to
same-species neighbors within `cooperation_range` (Chebyshev distance, default 2):

```python
shareable  = max(0.0, net_energy_gain_this_step)   # this step's catch only
donation   = cooperation_rate * shareable
recipients = same_species_agents_within(position, cooperation_range)
if recipients:
    split donation equally among recipients
    donor_energy -= donation
else:
    donation stays with donor (no partner available → no cost)
```

This is a **meal-sharing** model, not a continuous tax on standing energy. An agent that
doesn't eat this step donates nothing regardless of its genome value — donation is tied
to the flow (this step's gain), not the stock (accumulated energy). Donation only happens
on a subset of steps (whichever ones include a catch), and only when an eligible neighbor
happens to be present.

## Why this can create a real Darwin/Baldwin loop, not just a cost sink

Naively, sharing energy with an unrelated stranger is a pure fitness cost with no
compensating benefit — indiscriminate altruism should always be selected to zero. What
makes `cooperation_rate` interesting here is a fact already built into this environment:
**offspring spawn adjacent to their parent** (`_find_available_spawn_position` tries
adjacent cells before falling back to a random free cell). This is *population viscosity* —
spatial neighbors are more likely to be kin than a random draw from the population, without
any explicit kin-recognition code. This is the standard substrate for kin selection
(Hamilton 1964): donations that are blind to kinship are nonetheless kin-biased, because kin
cluster in space by default.

That gives a genuine two-way coupling between genome and policy:

```
Direction 1 (Darwin):
  RL-learned dispersal behavior → how far agents drift from their birth site
                                → how related a typical spatial neighbor actually is
                                → whether cooperation_rate pays off (Hamilton's rule: rB > C)
                                → selection propagates the adaptive rate through the population

Direction 2 (Baldwin):
  genome (cooperation_rate) → changes local energy pooling every step
                            → changes the competitive/energetic landscape
                            → changes what the RL policy must learn (e.g. whether staying
                              near kin is worth it, whether isolation is safer)
```

Unlike `metabolic_rate` (see the eco-evolutionary metabolic-rate module), which needed an
artificial sub-linear/linear asymmetry (`gain = food^α`, α < 1) to avoid racing to its upper
bound, `cooperation_rate`'s individual-level tradeoff falls out of the ecology directly: some
neighbors are kin, some aren't, and the donor can't tell which in advance. The interior
optimum — if one exists — is set by the population's effective local relatedness, which is
itself an emergent, policy-dependent quantity.

## The open scientific risk: Taylor's cancellation

There is a known result in kin-selection theory (Taylor 1992) that in simple
limited-dispersal models, the *benefit* of local altruism to kin is often exactly cancelled
by the increased *local competition* among those same kin for the same limited resources and
spawn slots. Helping a sibling survive and reproduce can simply mean that sibling out-competes
your own future offspring for the same nearby food and space.

Whether `cooperation_rate` evolves positive, stays at its neutral founder value, or drifts
negative-pressure-bound (mutation pushing toward 0 with no compensating benefit) is a genuine
open question for this environment — not a guaranteed outcome. A null result (no drift from
0.0) is not necessarily a broken mechanism; it may be evidence that kin competition is
cancelling the kin-selection benefit here, which is itself a real and interesting finding.

## Tradeoffs between trait designs

| | `offspring_investment_fraction` | `metabolic_rate` (asymmetric, α<1) | `cooperation_rate` |
|---|---|---|---|
| Darwin/Baldwin loop | one-way (incomplete) | two-way (complete) | two-way, contingent on emergent dispersal |
| Individual stabilizing force | yes (interior optimum) | yes (digestive saturation) | uncertain — depends on relatedness vs. kin competition |
| Domain overlap with RL | none (reproduction only) | full (every step) | full, but only on eating steps |
| Requires spatial mechanism | no | no | yes (relies on existing spawn-near-parent viscosity) |
| Ease of training | easier | harder | hardest — outcome itself is the experiment |
| Interpretability | high | moderate | moderate — needs the relatedness proxy metric to interpret |

## Founder distribution

Starting at `mean=0.0` makes the initial population behave identically to the no-genome
baseline — no donation occurs until mutation introduces variation. Any drift away from 0.0
is therefore a direct signal that selection is acting:

```python
"cooperation_rate_mean": 0.0,   # neutral start — matches no-genome baseline exactly
"cooperation_rate_std":  0.05,  # initial variation for mutation/selection to explore
"trait_bounds": (0.0, 1.0),
"cooperation_range": 2,         # Chebyshev radius for donation eligibility
```

## What to watch in training

- `eco_evolution/{species}_cooperation_rate_mean` — drift from 0.0 indicates selection;
  positive drift means cooperation is being favored, and given the founder mean is the
  lower bound, only positive drift is structurally possible
- `eco_evolution/{species}_cooperation_rate_std` — narrowing means selection is
  concentrating the population around a particular value; flat std at ~mutation-equilibrium
  means the trait is currently neutral
- `eco_evolution/{species}_local_relatedness_proxy` — fraction of donated energy (population-
  wide, per episode) that went to genuine kin (parent, offspring, or full sibling, from the
  existing `agent_parents` lineage) vs. unrelated same-species neighbors. This is the
  moderator variable: high relatedness proxy is the condition under which Hamilton's rule
  favors cooperation; low relatedness proxy means most donations are wasted on strangers
- `eco_evolution/{species}_energy_donated_total` / `_energy_received_total` — population-level
  flow sanity check (should be equal, since donations transfer within the population)
- `eco_evolution/{species}_coop_repro_spearman` — Spearman correlation between individual
  `cooperation_rate` and whether that agent reproduced (binary). Positive = cooperation pays
  off; negative = cooperation is a net cost (mirrors `metabolic_rate`'s `mr_repro_spearman`).
  A shift in this correlation that tracks a shift in `local_relatedness_proxy` — rather than
  in policy quality per se — is the signature to look for: it points at dispersal behavior
  (not foraging skill) as the driver
- `eco_evolution/{species}_coop_repro_rate_q1` through `_q4` — reproduction fraction per
  cooperation-rate quartile. A monotone Q1→Q4 gradient shows a clean direction; if
  `cooperation_rate` never departs from ~0, all quartiles will look identical — evidence
  the trait stayed neutral

## Confirming the reverse leg (genome → RL)

As with `metabolic_rate`, the reverse leg (does the genome shift feed back into what the
policy needs to learn?) is the hardest direction to establish. The same natural-variation
approach applies: mutation guarantees within-episode variation in `cooperation_rate`, so
`{species}_coop_repro_spearman` can be tracked against `{species}_local_relatedness_proxy`
and `{species}_spawned_total` without needing a frozen-genome controlled experiment. The
distinguishing signature for this trait specifically: if the reverse leg is real, an increase
in `local_relatedness_proxy` (driven by the policy learning tighter or looser dispersal,
independent of raw foraging competence) should predict a shift in `coop_repro_spearman`,
whereas a `metabolic_rate`-style effect is driven by foraging competence
(`spawned_total`) instead.

## Relationship to the Darwin/Baldwin process

The Baldwin Effect requires:

1. Individual learning discovers a beneficial behavior within a lifetime
2. That discovery creates selective pressure on heritable traits
3. Over generations, the trait that made learning easier (or substituted for it) becomes
   fixed in the genome

For `cooperation_rate`, step 1 is indirect: the "behavior" the policy discovers isn't
cooperation itself (donation is automatic, not a learned action) — it's **dispersal
distance**. The policy's emergent movement pattern determines local relatedness, which in
turn determines whether the genome's cooperation_rate is currently adaptive. This is a
subtler version of the Baldwin loop than `metabolic_rate`'s: the RL policy shapes the
*context* in which the genome trait is evaluated, rather than directly competing with the
genome trait for the same resource.

This makes `cooperation_rate` a genuinely different experiment from `metabolic_rate`, not
just a variation on the same theme: it tests whether an *indirect*, context-setting
Baldwinian channel (dispersal → relatedness → genome fitness) can close a Darwin/Baldwin
loop, versus `metabolic_rate`'s *direct* channel (genome value competes with policy quality
for the same energy budget every step).

## Observation visibility: `cooperation_rate` is currently policy-blind

The observation space is a fixed 3-channel grid (`num_obs_channels: 3`: predator-energy,
prey-energy, grass-energy — see `_build_observation_space` / `_get_observation`). Neither
an agent's own `cooperation_rate` nor a neighbor's is exposed anywhere in the observation.
Donation is computed mechanically in `_apply_cooperative_donation`
(`donation = cooperation_rate * shareable`) every qualifying step, with no possibility for
the policy to condition behavior on the trait — its own or anyone else's.

Practically, this means the only thing the policy can learn is **dispersal** (whether to
stay near kin or roam), not a *conditional* cooperation strategy (e.g., "be generous near
high-cooperation neighbors, withhold near strangers"). The current design therefore tests
only **unconditional, viscosity-based kin selection** (Hamilton's original mechanism:
blind donation + spatial relatedness from spawn-near-parent), not the plasticity/reputation
-based Baldwin-effect-for-cooperation mechanism described in the Suzuki & Arita line of
work, which requires the agent to sense something about its partner and adapt in real
time. A null or negative drift result here would rule out unconditional kin selection in
this ecology, but would *not* rule out that broader class of mechanism.

## Possible follow-up: trait-based assortment ("green-beard" effect)

Making a neighbor's `cooperation_rate` visible in the observation (e.g., an extra grid
channel populated at neighbor positions) would open a second, distinct pathway to
cooperation: **trait-based assortment**, also known as the **green-beard effect** (Dawkins
1976) or tag-based cooperation (Riolo, Cohen & Axelrod 2001). Instead of cooperation being
favored because neighbors are *genetically related* (kin selection), it would be favored
because the policy learns to preferentially move toward/stay near *visibly
high-cooperation* neighbors regardless of relatedness — the trait itself acts as its own
recognition tag.

**What "green-beard" means here, explicitly.** Dawkins' original thought experiment posited
a single gene that does two things at once: it gives its carrier a visible marker (a green
beard) *and* makes the carrier behave cooperatively toward anyone else displaying that same
marker. Cooperation can then spread between total strangers, because the tag lets
cooperators find and favor each other — no genetic relatedness required, just recognizable
similarity in the one trait that matters.

Mapped onto this environment: if `cooperation_rate` were visible to neighbors, a
high-`cooperation_rate` agent could learn (via the RL policy) to preferentially move toward
and stay near other high-`cooperation_rate` agents. That clustering is **positive
assortment** — cooperators end up disproportionately surrounded by other cooperators, so
donations mostly return to individuals who reciprocate the pattern, while low-cooperation
agents end up isolated among each other, denied access to the group's shared catch bonus.
That can make cooperation self-reinforcing *without any kinship at all* — a pathway
entirely separate from the kin-selection mechanism this module currently tests (which
relies on spawn-adjacent-to-parent viscosity, not trait recognition).

One structural point in favor of testing this here specifically: the standard real-world
objection to green-beard genes is the "false-beard" problem — a mutant could evolve the
visible tag without the costly cooperative behavior, exploiting the recognition system as a
free rider. That failure mode is structurally harder to get in this environment, because
`cooperation_rate` isn't a separate tag-plus-behavior pair — the same scalar *is* both the
signal and the donation amount. An agent can't display "I'm a 0.8 cooperator" while only
actually donating at 0.1; the number that would be visible is the exact number that
determines the donation. That sidesteps one of the standard theoretical weaknesses of
green-beard mechanisms, which is worth keeping in mind if this follow-up is ever built.

This is a legitimate and interesting mechanism, but it is a **different experiment**, not a
tweak to this one:

- Own-trait visibility is of limited value on its own — `cooperation_rate` is fixed for an
  agent's entire lifetime (set at birth, only varying across generations via mutation), so
  the policy could at most learn a static bias from it, not an adaptive in-lifetime response.
- Neighbor-trait visibility is the interesting case, but it introduces a second candidate
  explanation for any observed drift. If `cooperation_rate` evolves positive after adding
  this, it would no longer be possible to tell whether the driver is spatial kin viscosity
  (this module's current hypothesis) or learned assortment, without running both variants
  side by side.

Recommendation: keep this module's observation space as is to preserve it as a clean,
single-mechanism test of kin selection. If assortment is worth testing, build it as a
separate variant (own module or a config flag) and compare its genome-drift trajectory
against this one, rather than conflating both mechanisms in a single run.
