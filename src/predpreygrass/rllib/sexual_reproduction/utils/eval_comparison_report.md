# Eval Comparison Report

Generated: 2026-02-03 20:22:04

Source files:
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/sexual_reproduction/ray_results/eval_comparison.csv`
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/sexual_reproduction/ray_results/eval_comparison_summary.csv`

Summary (aggregated across eval dirs)

| scavenger | n_eval_dirs | n_runs_total | join_rate | defect_rate | coop_capture_rate | free_rider_share | failure_rate | coop_fail_rate | solo_fail_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 1 | 10.0000 | 0.8973 | 0.1027 | 0.6469 | 0.0127 | 0.8848 | 0.7288 | 0.9439 |
| 0.2000 | 1 | 10.0000 | 0.9415 | 0.0585 | 0.5611 | 0.0108 | 0.9045 | 0.7731 | 0.9452 |
| 0.3000 | 1 | 10.0000 | 0.6183 | 0.3817 | 0.4748 | 0.0840 | 0.8883 | 0.6729 | 0.9300 |
| 0.4000 | 1 | 10.0000 | 0.6559 | 0.3441 | 0.4004 | 0.0809 | 0.8350 | 0.5224 | 0.8852 |

Failure rate by prey type

| scavenger | mammoth_fail_rate | rabbit_fail_rate |
| --- | --- | --- |
| 0.1000 | 0.9187 | 0.1249 |
| 0.2000 | 0.9404 | 0.1887 |
| 0.3000 | 0.9270 | 0.0974 |
| 0.4000 | 0.9128 | 0.1427 |

Plots

### Join vs Defect Decision Rate
![Join vs Defect Decision Rate](../ray_results/eval_comparison_summary_plots/join_defect_rate.png)

### Coop vs Solo Capture Rate
![Coop vs Solo Capture Rate](../ray_results/eval_comparison_summary_plots/coop_solo_capture_rate.png)

### Free Rider Share
![Free Rider Share](../ray_results/eval_comparison_summary_plots/free_rider_share.png)

### Team Capture Failure Rate
![Team Capture Failure Rate](../ray_results/eval_comparison_summary_plots/team_capture_failure_rate.png)

### Failure Rate: Coop vs Solo
![Failure Rate: Coop vs Solo](../ray_results/eval_comparison_summary_plots/team_capture_failure_rate_coop_solo.png)

### Failure Rate: Mammoth vs Rabbit
![Failure Rate: Mammoth vs Rabbit](../ray_results/eval_comparison_summary_plots/team_capture_failure_rate_prey.png)

### Free Rider Rates in Coop Captures
![Free Rider Rates in Coop Captures](../ray_results/eval_comparison_summary_plots/coop_free_rider_rates.png)

