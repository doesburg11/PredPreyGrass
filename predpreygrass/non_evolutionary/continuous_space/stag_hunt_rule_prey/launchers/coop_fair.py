"""Cooperation against chance with lean and fat rabbits, counting each approach
once. An approach starts when a hunter touches a rabbit it has not touched in
the last GAP physics steps; repeated touches within GAP (retries after a failed
attempt) belong to it. For each approach: was another hunter within the party
radius of the rabbit, and was the rabbit fat (energy >= rabbit_fat_energy) or
lean (<= rabbit_lean_energy)? Chance: the share of hunter time with another
hunter within the party radius. Hunters in a catch cooldown count too.

Cooperation is shown if approaches to fat rabbits have a partner near more
often than approaches to lean rabbits and than chance (RESULTS.md, section 12).

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_rule_prey/launchers/coop_fair.py \\
        CHECKPOINT EPISODES WORKERS
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

GAP = 64  # physics steps
_w = {}


def init(checkpoint):
    import torch

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.checkpoints import checkpoint_env_config, load_modules
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import StagHuntRulePrey

    torch.set_num_threads(1)
    config = checkpoint_env_config(checkpoint)
    config["truncate_on_prey_extinction"] = False
    _w["modules"] = load_modules(checkpoint)
    _w["env"] = StagHuntRulePrey(config)


def play(seed):
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import evaluate

    env = _w["env"]
    radius = env.party_radius
    counts = {"hunter_steps": 0, "with_partner": 0}
    for kind in ("fat", "middle", "lean"):
        counts[f"{kind}_approaches"] = 0
        counts[f"{kind}_with_partner"] = 0
    last_touch = {}
    original_step, original_contacts = env._physics_step, env._contacts

    def step(moves):
        hunters = env.predators
        counts["hunter_steps"] += len(hunters)
        for h in hunters:
            if any(o is not h and env.torus.distance(h.position, o.position) <= radius for o in hunters):
                counts["with_partner"] += 1
        return original_step(moves)

    def contacts(animal):
        if animal.species == "rabbit":
            now = env.time_step
            near = [p for p in env.predators if env.torus.distance(animal.position, p.position) <= radius]
            for p in env.predators:
                if not env._touching(animal, p):
                    continue
                key = (p.id, animal.id)
                if now - last_touch.get(key, -GAP - 1) > GAP:
                    e = animal.energy
                    kind = "fat" if e >= env.fat_energy else "lean" if e <= env.lean_energy else "middle"
                    counts[f"{kind}_approaches"] += 1
                    counts[f"{kind}_with_partner"] += any(o is not p for o in near)
                last_touch[key] = now
        return original_contacts(animal)

    env._physics_step, env._contacts = step, contacts
    try:
        evaluate.play(env, _w["modules"], seed)
    finally:
        env._physics_step, env._contacts = original_step, original_contacts
    return counts


def main():
    checkpoint, episodes, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init, initargs=(checkpoint,)) as pool:
        rows = list(pool.map(play, range(episodes)))
    t = {k: sum(r[k] for r in rows) for k in rows[0]}
    chance = t["with_partner"] / t["hunter_steps"]
    print(checkpoint)
    print(f"  chance (hunter time with a partner near): {chance:.3f}")
    share = {}
    for kind in ("lean", "middle", "fat"):
        n = t[f"{kind}_approaches"]
        share[kind] = t[f"{kind}_with_partner"] / n if n else float("nan")
        print(f"  {kind:6s} approaches {n:6d}, with a partner near {share[kind]:.3f} ({share[kind] / chance:.2f}x chance)")
    fat_above = sum(
        r["fat_approaches"] and r["lean_approaches"] and r["fat_with_partner"] / r["fat_approaches"] > r["lean_with_partner"] / r["lean_approaches"]
        for r in rows
    )
    print(f"  episodes with fat share above lean share: {fat_above} of {len(rows)}")
    print(f"  cooperation shown (fat > lean and fat > chance): {share['fat'] > share['lean'] and share['fat'] > chance}")


if __name__ == "__main__":
    main()
