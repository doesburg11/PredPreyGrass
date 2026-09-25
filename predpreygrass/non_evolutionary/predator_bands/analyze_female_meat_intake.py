"""
How little meat can a female take in and still survive? From rollouts of trained band policies (same instrumentation as
analyze_band_behavior.py) this reports the distribution of each female's NET meat intake per step of life,
    (own hunting + received via band sharing / care / gifts - given away) / life length,
for all females, for females that lived at least --min-life steps, and for females still alive when the episode ended (censored:
they survived the whole episode). The rules give a floor for comparison: meat drains at diet_meat_cost_share x (homeostatic +
move cost) per step, i.e. 0.010 (always idle) to 0.018 (always moving) per step at meat share 0.10, from a starting store of
2.5 energy; births add 0.9 x 5.0 x meat_share per birth. Descriptive; intake is not the same as consumption (energy left in the
store at the end of a life is not subtracted, and shares given away are).

Example:
  python -m predpreygrass.non_evolutionary.predator_bands.analyze_female_meat_intake \
      --run MEAT060_s42=~/simulation_results/ray_results/PPO_PREDATOR_BANDS_CALIB_F008_MEAT060_SEED42 --checkpoint 19 --episodes 20
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_bands.analyze_band_behavior import load_modules, rollout


def net_rate(lives):
    return np.array([(l["own_meat"] + l["recv_meat"] - l["given_meat"]) / max(l["life"], 1) for l in lives])


def describe(name, lives):
    if not lives:
        return f"   {name:34} n=0"
    r = net_rate(lives)
    q = np.percentile(r, [0, 5, 25, 50, 75])
    return (
        f"   {name:34} n={len(lives):5d} net meat/step: min {q[0]:+.4f}  p5 {q[1]:+.4f}  p25 {q[2]:+.4f}  median {q[3]:+.4f}  p75 {q[4]:+.4f}"
        f" | mean life {np.mean([l['life'] for l in lives]):6.1f}, mean own meat {np.mean([l['own_meat'] for l in lives]):5.2f}, "
        f"mean received {np.mean([l['recv_meat'] for l in lives]):5.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=19)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min-life", type=int, default=300)
    parser.add_argument("--seed", type=int, default=6000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        _, lives = rollout(config, modules, args.episodes, args.seed)
        fem = [l for l in lives if l["sex"] == "predator_female"]
        mal = [l for l in lives if l["sex"] == "predator_male"]
        share = config.get("diet_meat_cost_share")
        floor = (share * config["homeostatic_energy_cost_per_step_predator"],
                 share * (config["homeostatic_energy_cost_per_step_predator"] + config["move_energy_cost_per_step_predator"]))
        print(f"== {label} ckpt{args.checkpoint}: meat share {share}, rule floor {floor[0]:.4f} (idle) to {floor[1]:.4f} (always moving) per step", flush=True)
        print(describe("females, all lives", fem))
        print(describe(f"females, lives >= {args.min_life} steps", [l for l in fem if l["life"] >= args.min_life]))
        print(describe("females, lives >= 800 steps", [l for l in fem if l["life"] >= 800]))
        print(describe("females alive at episode end", [l for l in fem if l["censored"]]))
        print(describe("males, all lives (comparison)", mal))
        print(flush=True)


if __name__ == "__main__":
    main()
