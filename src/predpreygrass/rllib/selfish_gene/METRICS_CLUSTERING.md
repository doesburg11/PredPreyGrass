# Measuring Cooperation under Selfish Gene
We never hardcode helping; we detect it via lineage outcomes.

Metrics per role (pred/prey), per window W:
1) Kin clustering: mean distance to nearest same-lineage; same-lineage neighbor fraction in radius R.
2) Does clustering pay?: corr(same-lineage fraction, survival/offspring); lineage Δshare vs null (expected from starting share).
3) Controls: shuffle lineage tags within role; fixed-seed eval (no learning).
Plots: lineage survival curves; kin-distance histograms; clustering→fitness scatter.
