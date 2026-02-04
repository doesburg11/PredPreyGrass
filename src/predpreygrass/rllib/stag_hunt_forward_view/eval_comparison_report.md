# Eval Comparison Report

Generated: 2026-02-03 23:38:42

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

Join cost summary

| scavenger | join_cost_total | join_cost_events | predators_with_cost | predators_total | join_cost_per_event | join_cost_per_predator | join_cost_per_predator_all |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 839.7200 | 41986.0000 | 5597.0000 | 6779.0000 | 0.0200 | 0.1500 | 0.1239 |
| 0.2000 | 790.8000 | 39540.0000 | 4802.0000 | 5610.0000 | 0.0200 | 0.1647 | 0.1410 |
| 0.3000 | 486.6400 | 24332.0000 | 5062.0000 | 6620.0000 | 0.0200 | 0.0961 | 0.0735 |
| 0.4000 | 395.1600 | 19758.0000 | 4058.0000 | 5829.0000 | 0.0200 | 0.0974 | 0.0678 |

Join cost per successful capture

| scavenger | join_cost_per_successful_capture | join_cost_per_coop_capture |
| --- | --- | --- |
| 0.1000 | 0.1758 | 0.1912 |
| 0.2000 | 0.1976 | 0.2176 |
| 0.3000 | 0.1060 | 0.1527 |
| 0.4000 | 0.0935 | 0.1252 |

Plots

### Join vs Defect Decision Rate
![Join vs Defect Decision Rate](../../../../assets/eval_comparison_summary_plots/join_defect_rate.png)

### Coop vs Solo Capture Rate
![Coop vs Solo Capture Rate](../../../../assets/eval_comparison_summary_plots/coop_solo_capture_rate.png)

### Free Rider Share
![Free Rider Share](../../../../assets/eval_comparison_summary_plots/free_rider_share.png)

### Team Capture Failure Rate
![Team Capture Failure Rate](../../../../assets/eval_comparison_summary_plots/team_capture_failure_rate.png)

### Failure Rate: Coop vs Solo
![Failure Rate: Coop vs Solo](../../../../assets/eval_comparison_summary_plots/team_capture_failure_rate_coop_solo.png)

### Failure Rate: Mammoth vs Rabbit
![Failure Rate: Mammoth vs Rabbit](../../../../assets/eval_comparison_summary_plots/team_capture_failure_rate_prey.png)

### Free Rider Rates in Coop Captures
![Free Rider Rates in Coop Captures](../../../../assets/eval_comparison_summary_plots/coop_free_rider_rates.png)

### Join Cost per Predator
![Join Cost per Predator](../../../../assets/eval_comparison_summary_plots/join_cost_per_predator.png)

### Join Cost per Successful Capture
![Join Cost per Successful Capture](../../../../assets/eval_comparison_summary_plots/join_cost_per_capture.png)

