# PredPreyGrass — Selfish Gene: Searching for Emergent Cooperation

This document summarizes an investigation into the emergence (or absence) of cooperation in a minimal multi‑agent predator–prey–grass environment. The focus was to avoid “pre‑engineering” cooperative outcomes and instead observe whether cooperation is learned from basic pressures.

We started with the existing walls_occlusion setup and progressed to a “selfish_gene” variant with lineage‑aware reward and observation options, alongside offline metrics to quantify cooperation signals.


## Guiding principles
- Minimal augmentation first: reduce hand‑crafted incentives to avoid pre‑determined goal seeking.
- Seek fundamental drivers: if cooperation exists here, it should surface from selection pressures, not bespoke reward shaping.
- Measure learning, not imposition: any cooperation should be learned by agents, not baked into the objective.


## Environment lineage and variants
- Baseline: `walls_occlusion` version of PredPreyGrass (gridworld with line‑of‑sight occlusion and static walls). Only movement actions are available to agents.
- Selfish Gene variant: adds lineage tracking and optional lineage‑related observation channels without changing the action space.
- Training: PPO (RLlib new API) with per‑policy modules and evaluation from checkpoints.


## Minimalist step 1: lineage reward (Selfish Gene)
- Replace direct reproduction reward with a windowed lineage reward: reward depends not only on an agent’s own reproduction but also on the survival of its recent descendants over a window. This approximates the “selfish gene” perspective and provides a lightweight path to kin selection signals without directly rewarding cooperation.
- Rationale: If kin selection yields advantages (e.g., clustering, escorting kin into resource patches), lineage‑level returns should reflect that.
- Constraint: Action space remains movement only. Cooperation would have to manifest as spatial assortment (clustering) or coordinated movement, not through explicit transfers.

Outcome in this phase: no clear increase in learned clustering could be evidenced (see metrics below). Lineage reward alone did not cause rising kin clustering across training checkpoints.


## Step 2: extend observation (still minimal)
- Add optional observation‑only “kin‑density” channels so agents can perceive local density of same‑lineage conspecifics; include a line‑of‑sight (LOS) aware variant.
- Keep evaluation and training aligned for observation‑critical flags (LOS, masking, kin‑density radius, normalization cap). Observations were extended without adding new actions or direct “help” mechanics.

Outcome: even with kin‑density observations available, post‑hoc metrics did not show increasing learned clustering.


## Metrics and offline evaluation
We built a small pipeline to quantify cooperation proxies offline, independent of reward shaping:

- Assortment Index (AI): measures how often agents are adjacent to kin relative to a shuffled baseline (AI ≈ 0 means random assortment; higher is more kin clustering). LOS‑aware counting available.
- Kin Proximity Advantage (KPA): measures whether being near kin correlates with higher reproduction success (positive = advantage; negative = disadvantage). LOS‑aware option available.
- Per‑policy breakdowns: compute AI/KPA by policy (e.g., predator vs prey) for diagnosis.
- Bootstrap CIs: analysis script supports bootstrap confidence intervals; the evaluator uses no‑bootstrap for speed by default.

Artifacts and scripts
- Evaluator: `src/predpreygrass/rllib/selfish_gene/analysis/eval_checkpoints_coop.py` sweeps checkpoints, logs minimal per‑episode data, computes AI/KPA, and writes a summary CSV plus per‑checkpoint JSON.
- Plotter: `src/predpreygrass/rllib/selfish_gene/analysis/plot_coop_time_series.py` renders AI and KPA vs training iteration.


## Results (example run)
Sweeping 5 checkpoints from run `PPO_SELFISH_GENE_2025-10-02_21-31-07` produced the following trend:
- AI is clearly positive (≈ 0.24 down to ≈ 0.20), meaning kin clustering exists above random.
- However, AI declines across checkpoints, i.e., clustering weakens with training.
- KPA is slightly negative at all checkpoints and trends more negative (≈ −0.007 → −0.009), indicating being near kin correlates with marginally lower reproduction success (consistent with resource competition and/or predator pressure on clusters).

![AI & KPA vs training iteration](assets/images/readme/PPO_SELFISH_GENE_2025-10-02_21-31-07.png)

Interpretation
- There is meaningful kin assortment (AI ≈ 0.2 is non‑trivial), but the direction of change suggests agents are learning dispersion strategies under current pressures.
- Slightly negative KPA implies proximity costs (scramble competition for grass, visibility to predators) outweigh benefits of kin proximity—at least with movement‑only actions.


## Training‑time visibility (sanity)
- A lightweight training callback emits online proxies to TensorBoard (e.g., coop/ai_raw, coop/ai_los, coop/kpa). These are for live monitoring and mirrored under `custom_metrics/coop/*`.
- Evaluations with “random actions” can be used as smoke tests to ensure offline metrics produce non‑zero values without reward shaping.


## Conclusions (so far)
- With lineage reward and minimal observation extensions, we did not observe increasing learned kin clustering over training.
- Cooperation via spatial assortment exists (positive AI), but the learned trend is toward reduced clustering and a small reproduction disadvantage when near kin (negative KPA).


## Next steps and ideas
If the goal is to elicit learned cooperation rather than impose it, consider incremental, low‑risk changes that create room for cooperative advantages while preserving minimalism:

1) Expand the action space slightly
- Energy sharing or partial harvest: allow agents to forgo consuming all available grass/prey immediately or transfer a fraction of energy to nearby kin. This introduces potential gains from reciprocity/kin support without hard‑coding rewards for it.
- Signaling: a cheap “ping” action that biases kin movement (still minimal, but opens coordination).

2) Adjust environment pressures (gentle nudges)
- Reduce proximity penalties: e.g., increase grass regrowth or cap predatory efficiency against clusters so that grouping isn’t strictly worse.
- Slightly increase vision radius so kin signals are usable at practical ranges.

3) Measurement upgrades
- Add bootstrap CIs to the evaluator CSV (ai_lo/ai_hi, kpa_lo/kpa_hi) for uncertainty bands over time.
- Continue per‑policy diagnostics to see which species drive (or resist) clustering.
- Optionally evaluate with stochastic action sampling (vs deterministic argmax) to reveal latent cooperative modes.

4) Keep version isolation
- Maintain lineage of experiment folders (`vX_Y`) and do not back‑port changes to older versions; ensure `run_config.json` snapshots are stored for reproducibility.


## How to reproduce the results
1) Evaluate checkpoints
```
python src/predpreygrass/rllib/selfish_gene/analysis/eval_checkpoints_coop.py \
  --run ~/Dropbox/02_marl_results/predpreygrass_results/ray_results/PPO_SELFISH_GENE_2025-10-02_21-31-07 \
  --episodes 20 \
  --max-steps 800 \
  --seed 0 \
  --limit 5 \
  --out output/coop_eval_summary.csv \
  --log-root output/coop_eval_logs \
  --progress-interval 50
```

2) Plot time series
```
python src/predpreygrass/rllib/selfish_gene/analysis/plot_coop_time_series.py \
  --csv output/coop_eval_summary.csv \
  --out output/coop_eval_plots/PPO_SELFISH_GENE_2025-10-02_21-31-07.png \
  --ema 0.3
```

3) Optional: bootstrap a single checkpoint for CIs
```
python src/predpreygrass/rllib/selfish_gene/analysis/coop_metrics.py \
  --log-dir output/coop_eval_logs/checkpoint_000005 \
  --los-aware \
  --bootstrap 2000
```


## Repository pointers
- Source env (selfish gene): `src/predpreygrass/rllib/selfish_gene/predpreygrass_rllib_env.py`
- Evaluator & metrics: `src/predpreygrass/rllib/selfish_gene/analysis/`
- Original occlusion baseline: `src/predpreygrass/rllib/walls_occlusion/`
- Assets (plots): `assets/images/readme/`


---
If something here is inaccurate or you’d like a deeper dive (e.g., enable stochastic evaluation, add CI columns to the CSV, or prototype a minimal energy‑sharing action), say the word and we’ll wire it up.