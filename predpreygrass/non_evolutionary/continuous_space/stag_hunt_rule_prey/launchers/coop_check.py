"""Cooperation against chance, lean and fat rabbits: share of catch attempts made
with another hunter within the party radius, against the share of hunter time
spent with another hunter within that radius (sampled every physics step).
Run from the repository root.
usage: launchers/coop_check.py CHECKPOINT EPISODES WORKERS"""

import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

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
    original = env._physics_step

    def step(moves):
        hunters = env.predators
        counts["hunter_steps"] += len(hunters)
        for h in hunters:
            if any(o is not h and env.torus.distance(h.position, o.position) <= radius for o in hunters):
                counts["with_partner"] += 1
        return original(moves)

    env._physics_step = step
    try:
        evaluate.play(env, _w["modules"], seed)
    finally:
        env._physics_step = original
    t = env.episode_hunt_stats or env.hunt_stats
    counts["party"] = t["rabbits_caught_party"]
    counts["alone"] = t["rabbits_caught_solo"] + t["rabbit_catch_failures"]
    return counts


def main():
    checkpoint, episodes, workers = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=init, initargs=(checkpoint,)) as pool:
        rows = list(pool.map(play, range(episodes)))
    total = {k: sum(r[k] for r in rows) for k in rows[0]}
    attempts = total["party"] + total["alone"]
    time_share = total["with_partner"] / total["hunter_steps"]
    attempt_share = total["party"] / attempts
    per_episode = sorted(r["party"] / max(1, r["party"] + r["alone"]) - r["with_partner"] / max(1, r["hunter_steps"]) for r in rows)
    above = sum(d > 0 for d in per_episode)
    print(f"{checkpoint}\n  attempts with a partner near: {attempt_share:.3f} ({total['party']} of {attempts})"
          f"\n  hunter time with a partner near (chance): {time_share:.3f}"
          f"\n  ratio {attempt_share / time_share:.2f}x; episodes with attempt share above time share: {above} of {len(rows)}")


if __name__ == "__main__":
    main()
