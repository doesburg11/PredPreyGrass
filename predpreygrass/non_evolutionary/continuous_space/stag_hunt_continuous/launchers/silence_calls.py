"""Exploratory: replay a signal checkpoint with the heard-call input zeroed
(policies unchanged), and measure the fair cooperation share (RESULTS section 8.1).

Run from the repository root:
    python predpreygrass/non_evolutionary/continuous_space/stag_hunt_continuous/launchers/silence_calls.py CHECKPOINT SILENCE(0/1) EPISODES OUT.json"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous import evaluate
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.launchers import coop_fair
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import CALL_SIZE


def init(checkpoint, silence):
    evaluate._init(checkpoint, {})
    if silence:
        evaluate._worker["env"]._heard = lambda listener: [0.0] * CALL_SIZE


def one(seed):
    env = evaluate._worker["env"]
    row = coop_fair.count(env, evaluate._worker["modules"], seed)
    row["end"] = env.time_step
    row["alive_predator"] = len(env.predators)
    row["alive_rabbit"] = len(env.rabbits)
    return row


if __name__ == "__main__":
    if sys.argv[2] not in ("0", "1"):
        raise SystemExit("SILENCE must be 0 or 1")
    ck, silence, episodes, out = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), sys.argv[4]
    with ProcessPoolExecutor(10, mp_context=get_context("spawn"), initializer=init, initargs=(ck, silence)) as pool:
        rows = list(pool.map(one, range(episodes)))
    s = coop_fair.summarise([{k: v for k, v in r.items() if k not in ("end", "alive_predator", "alive_rabbit")} for r in rows])
    s["both_alive_end"] = sum(r["alive_predator"] > 0 and r["alive_rabbit"] > 0 for r in rows)
    print(json.dumps(s))
    json.dump({"checkpoint": ck, "silence": silence, "summary": s, "episodes": rows}, open(out, "w"), indent=1)
