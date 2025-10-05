# Search for Learned Cooperation — Wrap‑Up

This short write‑up summarizes the project’s path to detecting learned cooperation, moving from walls/occlusion baselines to lineage signaling ("selfish_gene") and finally to kin_selection with explicit sharing.

## 1) Walls + Occlusion (baseline pressure)
- Code: `src/predpreygrass/rllib/walls_occlusion/predpreygrass_rllib_env.py`
- Goal: Create realistic spatial constraints (walls, LOS effects) and observe emergent behavior under basic energy dynamics. **Only sparse direct reproduction rewards**. 
- Outcome: Succesful propagation of Predator and Prey populations. Provided environmental structure but no clear cooperation signal on its own; behavior remained dominated by foraging/predation/reproduction dynamics.

## 2) "Selfish Gene" lineage‑only clustering attempt
- Code: `src/predpreygrass/rllib/selfish_gene/predpreygrass_rllib_env.py`
- Idea: Use lineage/offspring as a proxy to detect kin clustering and indirect cooperation without introducing an explicit cooperative action.
- Method: Added windowed lineage reward and tracked living offspring counts; looked for clustering and correlated success. **Disabled direct reproduction rewards (lineage‑only at birth)**,
- Outcome: Not particularly successful for gauging cooperation. While lineage counts were measurable, signals were weak/indirect and confounded by ecology (predation, resource layout). Clear learned cooperation did not surface consistently.

## 3) Kin selection + explicit SHARE (prey‑first)
- Code: `src/predpreygrass/rllib/kin_selection/predpreygrass_rllib_env.py`
- Key changes:
  - Kept lineage reward as a sparse, evolution‑aligned signal.
  - Introduced an explicit SHARE action (prey role first), LOS/radius/kin‑only eligibility, and cooldown/thresholds.
  - Exposed online metrics (helping_rate, share_attempt_rate, received_share_mean, per‑type variants, routing to same/other type) and offline evaluation/plots.
- Evidence of learned cooperation:
  - With lineage reward + SHARE enabled for the prey population in a type_1‑only setting, helping_rate ramped decisively and stabilized at meaningful levels in Tensorboard.

## 4) Refinements that made signals too sparse
- To minimize shaping and isolate evolutionary incentives, we:
  - Enforced LOS/eligibility thresholds and kin‑only routing (seperate type_1_prey and type_2_prey lineage rewards),
  - Avoided action masking for RLlib stability.
- Consequence: Signals became very sparse. Predator returns went near‑flat (reward only at reproduction), and helping dynamics showed thresholded step changes and occasional downshifts when eligibility/ecology shifted. Cooperation still present, but harder to read purely via episode returns.

## Conclusion
- Detected learned cooperation when:
  1) Adding lineage rewards (evolution‑aligned but sparse), and
  2) Adding an explicit SHARE action for the prey population (type_1‑only run), which produced clear, sustained increases in helping metrics.
  3) Further attempts to remove shaping entirely (lineage‑only at birth, no catch/eat reward, strict eligibility/LOS) led to signals that were too sparse for smooth learning curves, even though cooperation behavior could still emerge.
  4) Stopped this research route.

## Notes & pointers
- For readable curves without heavy shaping, consider re‑enabling a small, sparse reproduction reward (direct) while keeping catch/eat at 0; this maintains evolutionary focus but yields occasional positive returns.
- Key files: callbacks and analysis for online/offline metrics live under `src/predpreygrass/rllib/kin_selection/utils/` and `analysis/` respectively; trainer scripts wire the lineage/SHARE configs.
