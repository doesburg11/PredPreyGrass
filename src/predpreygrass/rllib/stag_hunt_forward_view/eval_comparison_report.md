# Eval Comparison Report

Generated: 2026-02-03 22:46:42

Source files:
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_forward_view/ray_results/eval_comparison.csv`
- `/home/doesburg/Projects/PredPreyGrass/src/predpreygrass/rllib/stag_hunt_forward_view/ray_results/eval_comparison_summary.csv`

Summary (aggregated across eval dirs)

| scavenger | n_eval_dirs | n_runs_total | join_rate | defect_rate | coop_capture_rate | free_rider_share | failure_rate | coop_fail_rate | solo_fail_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 1 | 10.0000 | 0.9024 | 0.0976 | 0.9196 | 0.0148 | 0.9132 | 0.7484 | 0.9898 |
| 0.2000 | 1 | 10.0000 | 0.9429 | 0.0571 | 0.9078 | 0.0108 | 0.9371 | 0.7873 | 0.9921 |
| 0.3000 | 1 | 10.0000 | 0.5944 | 0.4056 | 0.6940 | 0.1212 | 0.9228 | 0.7041 | 0.9711 |
| 0.4000 | 1 | 10.0000 | 0.6218 | 0.3782 | 0.7472 | 0.1109 | 0.9220 | 0.6133 | 0.9768 |

Failure rate by prey type

| scavenger | mammoth_fail_rate | rabbit_fail_rate |
| --- | --- | --- |
| 0.1000 | 0.9132 | 0.0000 |
| 0.2000 | 0.9371 | 0.0000 |
| 0.3000 | 0.9228 | 0.0000 |
| 0.4000 | 0.9220 | 0.0000 |

Plots

### Join vs Defect Decision Rate
![Join vs Defect Decision Rate](ray_results/eval_comparison_summary_plots/join_defect_rate.png)

### Coop vs Solo Capture Rate
![Coop vs Solo Capture Rate](ray_results/eval_comparison_summary_plots/coop_solo_capture_rate.png)

### Free Rider Share
![Free Rider Share](ray_results/eval_comparison_summary_plots/free_rider_share.png)

### Team Capture Failure Rate
![Team Capture Failure Rate](ray_results/eval_comparison_summary_plots/team_capture_failure_rate.png)

### Failure Rate: Coop vs Solo
![Failure Rate: Coop vs Solo](ray_results/eval_comparison_summary_plots/team_capture_failure_rate_coop_solo.png)

### Failure Rate: Mammoth vs Rabbit
![Failure Rate: Mammoth vs Rabbit](ray_results/eval_comparison_summary_plots/team_capture_failure_rate_prey.png)

### Free Rider Rates in Coop Captures
![Free Rider Rates in Coop Captures](ray_results/eval_comparison_summary_plots/coop_free_rider_rates.png)

