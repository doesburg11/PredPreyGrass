"""RESULTS.md sections 3.2 and 4: the deer's grass supply under logistic
grass for several root reserves (MODE root_reserve, section 3.2) or under
grazed grass for several stubble heights (MODE stubble, section 4), with
fixed behaviour: touch_1's final policies
(trained with instant grass) for hunters and deer, in the touch-rule world.
Under logistic grass the biomass input each animal gains is removed from its
observation, so the policies see exactly the inputs they were trained with.

Per setting and episode (seeds 0..EPISODES-1): grass energy gained by deer
per physics step, mean biomass per patch eaten, mean deer and hunter
numbers, whether hunters died out. "instant" is the reference.

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/grass_supply.py \\
        EPISODES WORKERS OUT.json [MODE]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
CHECKPOINT = RUNS / "touch_1" / "checkpoint"
ROOT_RESERVES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
STUBBLES = (0.05, 0.1, 0.2, 0.3, 0.4)
MODES = {
    "root_reserve": (ROOT_RESERVES, lambda v: {"grass_regrowth": "logistic", "grass_root_reserve": v}),
    "stubble": (STUBBLES, lambda v: {"grass_regrowth": "grazed", "grass_stubble": v}),
}


def episode(job):
    mode, setting, seed = job
    import torch

    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import evaluate
    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.checkpoints import checkpoint_env_config, load_modules
    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment import SELF_SIZE, SLOT_SIZE, SPECIES, DeerStagHunt

    torch.set_num_threads(1)
    config = checkpoint_env_config(CHECKPOINT)
    assert config["grass_regrowth"] == "instant" and config["deer_catch_rule"] == "touch"
    if setting != "instant":
        config.update(MODES[mode][1](setting))
    env = DeerStagHunt(config)
    modules = load_modules(CHECKPOINT)
    if setting != "instant":
        observe = env._observe
        biomass_at = {s: SELF_SIZE + SLOT_SIZE * sum(env.observe[(s, o)] for o in SPECIES) + 4 for s in SPECIES}

        def trimmed(animal):
            return np.delete(observe(animal), biomass_at[animal.species])

        env._observe = trimmed
    bites = []
    advance = env.grass.advance

    def counted(prey):
        eaten = advance(prey)
        bites.extend(eaten.values())
        return eaten

    env.grass.advance = counted
    counts = {"deer": 0, "predator": 0, "steps": 0}
    step = env._physics_step

    def counting(moves):
        counts["deer"] += len(env.deer)
        counts["predator"] += len(env.predators)
        counts["steps"] += 1
        return step(moves)

    env._physics_step = counting
    result = evaluate.play(env, modules, seed)
    steps = max(counts["steps"], 1)
    return {
        "setting": setting,
        "seed": seed,
        "physics_steps": result["physics_steps"],
        "grass_energy_per_step": sum(bites) * config["energy_grass_gain"] / steps,
        "biomass_per_bite": sum(bites) / len(bites) if bites else None,
        "mean_deer": counts["deer"] / steps,
        "mean_hunters": counts["predator"] / steps,
        "hunters_extinct": "predator" in result["extinct_at"],
    }


def main():
    if len(sys.argv) not in (4, 5) or (len(sys.argv) == 5 and sys.argv[4] not in MODES):
        raise SystemExit(__doc__)
    episodes, workers, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) == 5 else "root_reserve"
    candidates = MODES[mode][0]
    jobs = [(mode, s, seed) for s in ("instant",) + candidates for seed in range(episodes)]
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(episode, jobs))
    summary = {}
    for s in ("instant",) + candidates:
        mine = [r for r in rows if r["setting"] == s]
        summary[str(s)] = {key: float(np.mean([r[key] for r in mine if r[key] is not None])) for key in ("grass_energy_per_step", "biomass_per_bite", "mean_deer", "mean_hunters", "physics_steps")}
        summary[str(s)]["hunters_extinct"] = sum(r["hunters_extinct"] for r in mine)
    reference = summary["instant"]["grass_energy_per_step"]
    for s, v in summary.items():
        v["supply_ratio"] = v["grass_energy_per_step"] / reference
    chosen = min(candidates, key=lambda s: (abs(summary[str(s)]["supply_ratio"] - 1), s))
    print(f"{'setting':>8s} {'energy/step':>11s} {'ratio':>6s} {'biomass/bite':>12s} {'deer':>6s} {'hunters':>7s} {'steps':>6s} {'hunters extinct':>15s}")
    for s, v in summary.items():
        print(f"{s:>8s} {v['grass_energy_per_step']:11.3f} {v['supply_ratio']:6.2f} {v['biomass_per_bite']:12.2f} {v['mean_deer']:6.1f} {v['mean_hunters']:7.1f} {v['physics_steps']:6.0f} {v['hunters_extinct']:>8d}/{episodes}")
    print(f"chosen {mode.replace('_', ' ')} (supply ratio closest to 1, ties to the smaller): {chosen}")
    Path(out).write_text(json.dumps({"checkpoint": str(CHECKPOINT), "mode": mode, "summary": summary, "chosen": chosen, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
