# Risky Cooperation vs Safe Foraging in a Complex Ecosystem

## Research Question

This project does **not** test a clean, textbook stag hunt game.  
Instead, it asks a broader and ecologically grounded question:

> **In a complex ecosystem, under what minimal conditions do agents evolve from safe solo foraging toward risky cooperation, and how does voluntary defection shape that transition?**

More concretely, in the Predator–Prey–Grass (PPG) environment:

> **When do human predators prefer cooperating to hunt mammoths (risky, coordination-dependent) over solo hunting rabbits (safe, reliable), and when do they defect/free-ride instead?**

This framing preserves the full ecological richness of the environment while allowing controlled, interpretable experiments.

## Added Mechanics in This Variant

This `stag_hunt_forward_view` variant includes two changes relative to the base `stag_hunt` ecology:

- **Voluntary participation (defection):** predators choose `join_hunt` each step; only joiners contribute to capture.
- **Join cost and scavenging:** joiners pay `team_capture_join_cost` on successful captures; non-joiners can receive a small scavenger share.
- **Forward-view observations:** predator observations are shifted forward based on the last intended move (prey stay centered).

These additions create a richer social dilemma but also introduce a confound (observation change) that must be controlled in comparisons.

---

## Why This Is Not a Classical Stag Hunt

A classical stag hunt assumes:
- simultaneous choice
- static payoff matrix
- no population dynamics
- no spatial search costs
- no changing incentives over time

The PPG environment violates all of these assumptions:
- choices are **sequential and spatial**
- opportunities are **unevenly distributed**
- populations evolve during episodes
- energy, reproduction, and extinction change incentives dynamically

Therefore, the environment implements an **ecological stag-hunt analogue**, not a canonical game-theoretic stag hunt.

This is a *feature*, not a bug.

---

## Core Behavioral Trade-off

The environment instantiates the following trade-off:

| Strategy | Description | Properties |
|--------|------------|------------|
| **Safe foraging** | Solo hunting rabbits | Reliable, low payoff, low coordination cost |
| **Risky cooperation** | Coordinated mammoth hunting | High payoff, coordination-dependent, failure-prone |
| **Defection/free-riding** | Refuse to join, attempt to benefit from others | Low cost, can undermine cooperation |

The key question is **not** whether cooperation exists, but **when it becomes evolutionarily favored**.

---

## Minimal Conditions for Risky Cooperation

Rather than redesigning the environment, we isolate the *minimal ecological pressures* that shift behavior.

### 1. Payoff Gap (Necessity)
**Condition:**  
Expected long-term energy gain from successful mammoth cooperation must exceed rabbit foraging.

**Why it matters:**  
If mammoths are not clearly better *when successful*, cooperation is irrational.

**Knobs:**
- Mammoth energy yield
- Energy split rule
- Rabbit energy yield

---

### 2. Coordination Threshold (True Dilemma)
**Condition:**  
Solo mammoth hunting must usually fail, while pair (or group) hunting usually succeeds.

**Why it matters:**  
If solo success is common, the problem degenerates into individual optimization.

**Knobs:**
- Cumulative predator energy threshold
- Team-capture condition
- Required number of predators

---

### 3. Opportunity Structure (Choice Visibility)
**Condition:**  
Agents must frequently face situations where:
- rabbits are available **and**
- mammoth cooperation is feasible

**Why it matters:**  
Without overlapping opportunities, observed behavior reflects availability, not preference.

**Knobs:**
- Mammoth density
- Predator density / clustering
- Observation range

---

### 4. Cost of Coordination (Risk)
**Condition:**  
Cooperation must incur real costs (time, energy, exposure to starvation).

**Why it matters:**  
If cooperation is cheap, it is not risky and loses explanatory power.

**Knobs:**
- Per-step energy drain
- Movement cost
- Episode horizon

---

### 5. Credit Assignment (Stability)
**Condition:**  
Successful cooperation must not systematically reward free-riding or late arrival.

**Why it matters:**  
Perverse incentives can suppress stable cooperation even when payoffs are high.

**Knobs:**
- Energy split rule (equal vs proportional)
- Participant eligibility rules
- Join cost and scavenger share (defection incentives)

---

## Experimental Strategy: Minimal Intervention Ladder

To isolate causality, modify **one dimension at a time** while keeping the rest of the ecosystem intact.

### Stage A — Baseline Ecology
- Current environment settings
- Measure behavior without assumptions

### Stage B — Payoff Gap Only
- Increase mammoth payoff
- Keep all dynamics unchanged

### Stage C — Opportunity Structure Only
- Increase mammoth or predator encounter probability
- Keep payoffs unchanged

### Stage D — Coordination Threshold Only
- Tighten solo-failure / group-success boundary

### Stage E — Risk Reinforcement
- Increase coordination costs
- Test robustness of cooperation

This ladder answers: *what is the minimal pressure that causes a behavioral phase shift?*

### Stage F — Defection and Observation Ablation

To isolate the effects of defection vs observation changes, compare:

- `stag_hunt_defection` (defection only, centered view)
- `stag_hunt_forward_view` (defection + forward view)

This ablation separates social-dilemma effects from observation-shape effects.

---

## Measurement: What to Track (Not Episode Length)

Episode length is insufficient and misleading in ecological settings.

Instead, track **conditional behavior**:

### 1. Outcome-Based Metrics
- Solo rabbit captures
- Cooperative mammoth captures
- Failed attempts (intent signal)
- Join/defect rates per predator-step
- Free-rider exposure on successful captures

### 2. Opportunity-Normalized Rates
- Rabbit capture rate *given rabbit opportunity*
- Mammoth cooperation rate *given mammoth opportunity*
- Defection rate *given mammoth opportunity*

### 3. Simultaneous Opportunity Choice
> When both options are available, which is chosen next?

This is the strongest indicator of preference.

---

## Interpretable Hypothesis

A falsifiable hypothesis consistent with the environment:

> Risky cooperation emerges when  
> **(mammoth payoff × probability of timely partner availability)**  
> exceeds  
> **(rabbit payoff × probability of immediate success)**,  
> given that solo mammoth success is sufficiently unlikely.

This formulation explicitly separates:
- incentives
- coordination feasibility
- ecological risk

---

## Scientific Positioning

This environment does **not** answer:
> “Is stag hunt cooperation rational?”

It *does* answer:
> “Under what ecological pressures does cooperation emerge, stabilize, or collapse in spatial multi-agent populations?”

That makes it closer to:
- evolutionary game dynamics
- behavioral ecology
- open-ended MARL
than to static game theory.

---

## Summary

- The environment is **not a clean stag hunt**, but a **rich evolutionary analogue**
- Risky cooperation emerges only under specific, minimal ecological conditions
- These conditions can be isolated **without simplifying the environment**
- The result is behavior that is harder to analyze — but far more meaningful
