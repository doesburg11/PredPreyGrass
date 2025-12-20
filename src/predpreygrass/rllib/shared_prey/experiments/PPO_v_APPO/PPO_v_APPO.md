# PPO vs APPO in PredPreyGrass (PPG)

## Purpose

This document provides a **single, self-contained comparison** of **Proximal Policy Optimization (PPO)** and **Asynchronous Proximal Policy Optimization (APPO)** for the **PredPreyGrass (PPG)** environment. It is written to be directly usable in a repository, paper draft, or documentation site without any additional formatting or front-matter.

PredPreyGrass is a **multi-agent, population-based predator–prey–resource system** with:

* dynamic birth and death,
* energy-based survival and reproduction,
* long-horizon feedback loops,
* strong non-stationarity by design.

The key question addressed here is **not** “which algorithm learns faster in general,” but:

> *Which algorithm better preserves the meaning and interpretability of population-level dynamics in PPG?*

---

## PredPreyGrass assumptions

This comparison assumes the canonical properties of your environment:

* multiple agent roles (predators, prey; possibly multiple types),
* dynamically changing agent sets,
* policies that co-adapt over time,
* population persistence (avoiding extinction) as a core success criterion,
* Ray RLlib **new API stack** (Ray ≥ 2.40; you are on 2.48),
* reward signals used as *proxies* for longer-term ecological success.

---

## High-level takeaway

* **PPO** prioritizes *learning stability and interpretability*.
* **APPO** prioritizes *throughput and wall-clock speed*.
* In PredPreyGrass, this trade-off often changes the **observed ecology itself**.
* See detailed eval comparison: `comparison_PPO_vs_APPO_init_prey_2_5.md`

If PPG is treated as a **scientific simulator**, PPO is the safer default.
If PPG is treated as an **engineering throughput problem**, APPO can be useful.

---

## Algorithm overview

### PPO (Proximal Policy Optimization)

PPO is a **synchronous, on-policy** algorithm:

1. Collect a batch of trajectories using the current policy.
2. Perform several optimization epochs on that batch.
3. Discard the batch.
4. Repeat.

Key characteristics:

* minimal policy lag,
* tighter on-policy guarantees,
* coherent advantage estimation,
* easier debugging and reproducibility.

Policy updates are based on experience that reflects the *current* population state.

---

### APPO (Asynchronous PPO)

APPO is a **distributed, asynchronous** variant of PPO:

* rollout workers collect experience continuously,
* the learner updates the policy while rollouts are still running,
* training batches contain data from **slightly older policy versions**.

Key characteristics:

* much higher sample throughput,
* policy lag between acting and learning,
* increased sensitivity to rollout/learner timing,
* weaker causal attribution between update and outcome.

---

## Core conceptual difference

**PPO minimizes temporal mismatch between behavior and learning.
APPO accepts temporal mismatch to gain speed.**

In PredPreyGrass, temporal mismatch directly affects population dynamics.

---

## Interaction with ecological feedback loops

PPG is dominated by coupled feedback:

policy → behavior → encounters → births/deaths → population structure → selection pressure → policy

With **PPO**, this loop is relatively tight:

* behavior changes are reflected in the data before the next update,
* population-level consequences are visible to the learner quickly.

With **APPO**, the loop is loosened:

* learning updates may be based on outdated interaction regimes,
* corrections can arrive *after* irreversible population changes (e.g. extinction).

This can amplify instability in ways that look ecological but are algorithm-induced.

---

## Multi-agent non-stationarity

PredPreyGrass already contains genuine non-stationarity:

* predators and prey learn simultaneously,
* population sizes fluctuate,
* reproduction introduces new agents,
* death removes agents permanently.

APPO adds **algorithmic non-stationarity** on top:

* mixed-policy experience within a batch,
* drifting value baselines,
* blurred “who was I competing against?” signals.

The combined effect can turn structured co-adaptation into hard-to-interpret noise.

---

## Reward interpretation and fitness meaning

In PPG, rewards typically approximate:

* survival,
* reproduction,
* lineage continuation,
* sustained coexistence.

These are **delayed, population-level outcomes**.

Under PPO:

* reward timing is more coherent,
* changes in reward are easier to map to behavioral causes,
* “fitness-like” interpretations are more defensible.

Under APPO:

* reward may be optimized under lagged conditions,
* short-term exploitation can be favored,
* long-term ecological viability can be sacrificed unintentionally.

---

## Practical comparison (PPG-oriented)

| Dimension                      | PPO         | APPO        |
| ------------------------------ | ----------- | ----------- |
| Wall-clock throughput          | Medium      | High        |
| Policy lag                     | Low         | Medium–High |
| Stability of co-adaptation     | Higher      | Lower       |
| Training-induced extinctions   | Less common | More common |
| Debuggability                  | Easier      | Harder      |
| Reproducibility                | Higher      | Lower       |
| Interpretability of dynamics   | Stronger    | Weaker      |
| Suitable for scientific claims | Yes         | Risky       |

---

## When PPO is the right choice in PPG

Use **PPO** when you care about:

* stable predator–prey coexistence,
* interpreting oscillations as ecological phenomena,
* cooperation, sharing, or lineage mechanisms,
* long-horizon population persistence,
* clean ablations and publishable results.

Rule of thumb:

> If the *dynamics themselves* are part of your research claim, use PPO.

---

## When APPO makes sense in PPG

APPO is appropriate when speed matters more than interpretability.

Typical use cases:

1. **Pretraining primitives**

   * navigation,
   * basic foraging,
   * chase/escape reflexes.
2. **Frozen-opponent experiments**

   * one side learns while the other is fixed,
   * reduced non-stationarity.
3. **Engineering and stress testing**

   * large-scale sampling,
   * finding rare environment bugs,
   * scalability experiments.

Rule of thumb:

> If PPG is a *means* to faster policy learning, APPO can be justified.

---

## Failure modes to watch under APPO

Indicators that APPO is distorting the ecology:

* sudden population collapse absent under PPO,
* oscillations that scale with rollout parallelism,
* high sensitivity to rollout fragment length,
* large variance across random seeds,
* policies that score well short-term but destabilize the ecosystem.

Such effects should be treated as **algorithm artifacts until proven otherwise**.

---

## Experimental design guidance

For a fair PPO vs APPO comparison in PPG, keep constant:

* environment configuration,
* observation/action spaces,
* network architecture,
* total environment steps,
* evaluation protocol and metrics.

Vary **only** the training algorithm and its required distributed settings.

---

## Metrics that matter more than episodic return

For population-level analysis, track:

* time to first extinction,
* long-run mean population sizes,
* oscillation amplitude and period,
* reproduction counts per type,
* lineage survival (if logged),
* spatial or encounter-rate statistics.

These reveal regime changes that raw return often hides.

---

## Recommended hybrid strategy

A pragmatic approach that balances speed and validity:

1. Train basic competencies with **APPO**.
2. Freeze policies and validate population stability.
3. Continue long-horizon experiments with **PPO**.
4. Report results under synchronous PPO whenever making ecological claims.

---

## Bottom line

In PredPreyGrass, **PPO and APPO are not interchangeable knobs**.

* **PPO** preserves ecological coherence and interpretability.
* **APPO** increases speed but can fundamentally alter observed dynamics.

If PredPreyGrass is used as a **scientific ecosystem**, PPO should be your default, with APPO used deliberately and cautiously.
