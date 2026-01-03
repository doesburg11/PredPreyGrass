# Positioning the Project: MARL, Game Theory, and Ecology

## Overview

This project deliberately sits **outside** the standard categories of
- classical game theory,
- classical multi-agent reinforcement learning (MARL), and
- pure ecology or evolutionary modeling.

Instead, it explores how **learned behavior**, **ecological interaction**, and **population-level selection** jointly shape the emergence and stability of cooperation.

The goal is *not* equilibrium convergence or reward maximization, but **adaptive viability**:  
whether behaviors persist under ecological pressure while agents continue to learn.

---

## Unified Comparison Table

| Dimension | Game Theory | Classical MARL | Ecology / Evolution | **This Project (Hybrid)** |
|---------|------------|---------------|---------------------|---------------------------|
| Primary focus | Strategic choice | Policy optimization | Survival & reproduction | **Adaptive viability** |
| Agents | Rational decision-makers | Learning agents | Organisms | **Learning organisms** |
| Adaptation mechanism | Reasoning / equilibrium | Learning (gradients) | Selection (genes) | **Learning + selection** |
| Timescale of adaptation | Instant / abstract | Within lifetime | Across generations | **Both timescales** |
| Learning within lifetime | ❌ | ✅ | ⚠️ (rare/limited) | **✅ central** |
| Evolution / population change | ❌ | ❌ | ✅ | **✅ central** |
| Environment | Static | Usually static | Dynamic | **Dynamic & endogenous** |
| Payoffs / rewards | Fixed payoff matrix | Fixed reward function | Emergent fitness | **Emergent fitness + local rewards** |
| Meaning of success | Equilibrium | Reward convergence | Population persistence | **Persistence under learning** |
| Cooperation | Equilibrium strategy | Joint reward maximization | Viable behavior | **Survivable learned behavior** |
| Failure | Suboptimal choice | Low reward | Death / extinction | **Death + policy elimination** |
| Population size | Fixed | Fixed | Variable | **Variable** |
| Agent heterogeneity | Rare | Often avoided | Natural | **Intentional & functional** |
| Strategy space | Explicit | Parameterized policy | Implicit traits | **Emergent behaviors** |
| Stationarity | Fully stationary | Mostly stationary | Non-stationary | **Strongly non-stationary** |
| Credit assignment | Explicit | Explicit | Implicit | **Explicit + implicit** |
| What persists | Equilibria | Policies | Traits / lineages | **Behaviors & lineages** |

---

## Key Interpretive Remarks

### 1. Not Classical Game Theory
The project does not operate with:
- fixed payoff matrices,
- static incentives,
- or equilibrium selection as a success criterion.

The “game” itself changes as populations, resources, and behaviors change.  
As a result, classical equilibrium concepts do not meaningfully apply.

---

### 2. Not Classical MARL
While MARL algorithms (e.g. PPO) are used, the project diverges from classical MARL in crucial ways:

- Rewards do not fully define success.
- Learning does not converge to a stationary optimum.
- Population dynamics and extinction matter more than policy loss curves.

MARL here is a **mechanism**, not the object of study.

---

### 3. Not Pure Ecology or Evolution Either
Unlike most ecological or evolutionary models:

- Agents possess powerful within-lifetime learning.
- Behavior is highly plastic.
- Adaptation does not rely solely on slow genetic change.

Ecology filters and stabilizes *learned* behaviors rather than replacing learning.

---

### 4. Two Interacting Adaptive Layers

The system combines two normally separate processes:

**Fast timescale — Learning (MARL):**
- Policies adapt within a lifetime.
- Credit assignment is explicit.
- Enables behavioral flexibility.

**Slow timescale — Selection (Ecology/Evolution):**
- Agents die or reproduce.
- Lineages persist or vanish.
- Filters which learned behaviors survive.

Learning explores; ecology selects.

---

### 5. Meaning of Cooperation in This Project

- Cooperation is **not** defined as an equilibrium.
- Cooperation is **not** defined as joint reward maximization.

Instead:

> **Cooperation is a behavioral pattern that persists because it supports survival and reproduction under ecological constraints.**

Agents do not “choose” to cooperate; cooperation emerges when solo strategies become ecologically unstable.

---

## Conceptual Positioning

This project occupies a space that is largely unexplored by existing benchmarks:

- learning **and** selection
- internal viability **and** explicit learning
- non-stationary environments
- population-level outcomes

It is best described as:

> **An artificial behavioral ecology implemented through multi-agent reinforcement learning.**

---

## One-Sentence Summary

> **The project studies cooperation as a learned behavior that must also survive ecological and evolutionary pressure, rather than as an optimized strategy or fixed trait.**

This positioning distinguishes it simultaneously from:
- classical game theory,
- classical MARL benchmarks,
- and pure ecological or evolutionary models.

It is the interaction between these domains — not any one of them alone — that defines the contribution.
