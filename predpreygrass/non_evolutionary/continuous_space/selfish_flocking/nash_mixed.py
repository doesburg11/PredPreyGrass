"""Mixed TurnAway / SELFish populations, after Hahn et al. (ICAART 2020).

For every split ``k`` TurnAway + ``n - k`` SELFish prey, the script plays
fixed-length episodes (no termination on any catch), counts catches per prey,
and reports each strategy's payoff as mean frames survived per catch. The
game is symmetric, so a split ``k`` is a pure Nash equilibrium when no single
prey gains by switching strategy:

- a TurnAway prey does not gain by switching: payoff(TA | k) >= payoff(SELF | k - 1)
- a SELFish prey does not gain by switching: payoff(SELF | k) >= payoff(TA | k + 1)

Deviation from the paper: caught prey respawn at a uniformly random position
(this module's environment) rather than at the densest spot.

Run from the repository root:

    python -m predpreygrass.non_evolutionary.continuous_space.selfish_flocking.nash_mixed \
        --checkpoint selfish_flocking/runs/dqn_seed1/dqn_step_0500000.pt --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.environment import (
    DISCRETE_TURNS,
    SelfishFlockingEnv,
    Settings,
)
from predpreygrass.non_evolutionary.continuous_space.selfish_flocking.evaluate import (
    load_policy,
    turn_away_action,
)


def play_split(policy, turn_away_count, settings, seed, device):
    """Play one fixed-length episode for one split.

    Returns per-prey catch counts, per-prey sums and counts of completed
    survival spells (respawn or episode start to catch), and the TurnAway mask.
    """
    n = settings.prey_count
    is_turn_away = np.arange(n) < turn_away_count
    env = SelfishFlockingEnv(settings)
    catches = np.zeros(n, dtype=np.int64)
    spell_sum = np.zeros(n, dtype=np.int64)
    spell_start = np.zeros(n, dtype=np.int64)

    def actions_for(observations, agents):
        actions = np.empty(len(agents), dtype=np.int64)
        learned = [i for i, agent in enumerate(agents) if not is_turn_away[agent]]
        for i, agent in enumerate(agents):
            if is_turn_away[agent]:
                actions[i] = turn_away_action(observations[i], discrete=True)
        if learned:
            batch = torch.as_tensor(np.stack([observations[i] for i in learned]), device=device)
            with torch.inference_mode():
                actions[learned] = policy(batch).argmax(dim=1).cpu().numpy()
        return actions

    peers = list(range(1, n))
    size = np.array([settings.width, settings.height])
    observation = env.reset(seed=seed, focal=0)
    done = False
    while not done:
        focal_action = int(actions_for([observation], [0])[0])
        chosen = {}

        def peer_policy(obs):
            chosen["peers"] = actions_for(obs, peers)
            return chosen["peers"]

        before_positions = env.positions.copy()
        before_headings = env.headings.copy()
        observation, _, done, info = env.step(
            focal_action, peer_policy, discrete=True, terminate_on_focal_catch=False
        )
        if info["caught"]:
            # Caught prey are respawned at a random spot with a random
            # heading; everyone else lands exactly where its turn sends it.
            turns = DISCRETE_TURNS[np.concatenate(([focal_action], chosen["peers"]))]
            radians = np.radians(before_headings + turns)
            expected = (
                before_positions
                + settings.prey_speed * np.stack((np.cos(radians), np.sin(radians)), axis=1)
            ) % size
            caught = np.flatnonzero(np.any(np.abs(env.positions - expected) > 1e-9, axis=1))
            if len(caught) != info["caught"]:
                raise RuntimeError("could not identify caught prey")
            catches[caught] += 1
            spell_sum[caught] += env.steps - spell_start[caught]
            spell_start[caught] = env.steps
    return catches, spell_sum, is_turn_away


def exposure_payoff(catches, exposure):
    """Frames survived per catch: exposure over catches (inf if never caught).

    This is the censoring-aware rate estimator; spells still running when an
    episode ends contribute exposure without a catch.
    """
    return exposure / catches if catches else np.inf


def strategy_payoffs(record, frames, pick=None):
    """Pooled payoffs per strategy for one split, optionally over resampled episodes."""
    catches, spells = record["catches"], record["spell_sum"]
    if pick is not None:
        catches, spells = catches[pick], spells[pick]
    out = {}
    for name, members in (("turn_away", record["mask"]), ("selfish", ~record["mask"])):
        if not members.any():
            continue
        total = int(catches[:, members].sum())
        exposure = frames * catches.shape[0] * int(members.sum())
        out[name] = {
            "payoff": exposure_payoff(total, exposure),
            "catches": total,
            "mean_completed_spell": float(spells[:, members].sum() / total) if total else None,
        }
    return out


def deviation_checks(payoffs, n):
    """For each split k: is it stable against a single TurnAway or SELFish switch?"""
    checks = []
    for k in range(n + 1):
        ta_switch = (
            payoffs[k - 1]["selfish"]["payoff"] > payoffs[k]["turn_away"]["payoff"] if k > 0 else False
        )
        self_switch = (
            payoffs[k + 1]["turn_away"]["payoff"] > payoffs[k]["selfish"]["payoff"] if k < n else False
        )
        checks.append(
            {"turn_away_gains_by_switching": ta_switch, "selfish_gains_by_switching": self_switch}
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prey-count", type=int, default=10)
    parser.add_argument("--size", type=float, default=40.0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--frames", type=int, default=100_000, help="paper: 100,000")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--records-dir",
        type=Path,
        help="with --split: save that split's raw counts here; without: load all splits from here",
    )
    parser.add_argument("--split", type=int, help="play only this TurnAway count (for parallel runs)")
    args = parser.parse_args()
    if args.split is not None and args.records_dir is None:
        parser.error("--split needs --records-dir")
    if args.split is None and args.out is None:
        parser.error("--out is required unless --split is given")

    settings = Settings(
        width=args.size,
        height=args.size,
        prey_count=args.prey_count,
        neighbors=5,
        max_steps=args.frames,
    )
    policy = load_policy(args.checkpoint, "dqn", 3 * (2 + settings.neighbors), args.device)
    n = args.prey_count

    def record_path(k):
        return args.records_dir / f"split_{k:02d}.npz"

    def play(k):
        catches, spells = [], []
        mask = None
        for episode in range(args.episodes):
            c, sp, mask = play_split(policy, k, settings, args.seed + 1000 * k + episode, args.device)
            catches.append(c)
            spells.append(sp)
        return {"catches": np.stack(catches), "spell_sum": np.stack(spells), "mask": mask}

    if args.split is not None:
        if not 0 <= args.split <= n:
            parser.error(f"--split must be in [0, {n}]")
        record = play(args.split)
        args.records_dir.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename, so a merge never reads a partial record.
        temporary = record_path(args.split).with_suffix(".tmp.npz")
        np.savez(
            temporary,
            frames=args.frames,
            episodes=args.episodes,
            seed=args.seed,
            checkpoint=str(args.checkpoint),
            size=args.size,
            prey_count=n,
            **record,
        )
        temporary.replace(record_path(args.split))
        print(json.dumps({"n_turn_away": args.split, **strategy_payoffs(record, args.frames)}), flush=True)
        return

    records = []
    for k in range(n + 1):
        if args.records_dir is not None:
            saved = np.load(record_path(k))
            for key, expected in (
                ("frames", args.frames),
                ("episodes", args.episodes),
                ("seed", args.seed),
                ("checkpoint", str(args.checkpoint)),
            ):
                if saved[key].item() != expected:
                    raise ValueError(f"{record_path(k)} has {key}={saved[key].item()!r}, expected {expected!r}")
            # Older records predate the size/prey_count fields; the mask still pins n and k.
            if "size" in saved and saved["size"].item() != args.size:
                raise ValueError(f"{record_path(k)} has size={saved['size'].item()}, expected {args.size}")
            mask = saved["mask"]
            if mask.shape != (n,) or int(mask.sum()) != k or saved["catches"].shape != (args.episodes, n):
                raise ValueError(f"{record_path(k)} does not hold {args.episodes} episodes of {k} TurnAway in {n} prey")
            records.append({key: saved[key] for key in ("catches", "spell_sum", "mask")})
        else:
            records.append(play(k))
        print(json.dumps({"n_turn_away": k, **strategy_payoffs(records[-1], args.frames)}), flush=True)

    point = [strategy_payoffs(record, args.frames) for record in records]
    point_checks = deviation_checks(point, n)

    # Splits are independent runs: resample episodes within each split per draw.
    rng = np.random.default_rng(args.seed)
    stable_share = np.zeros(n + 1)
    draws = {k: {"turn_away": [], "selfish": []} for k in range(n + 1)}
    for _ in range(args.bootstrap):
        resampled = [
            strategy_payoffs(record, args.frames, rng.integers(args.episodes, size=args.episodes))
            for record in records
        ]
        for k, check in enumerate(deviation_checks(resampled, n)):
            stable_share[k] += not (check["turn_away_gains_by_switching"] or check["selfish_gains_by_switching"])
            for name, value in resampled[k].items():
                draws[k][name].append(value["payoff"])
    stable_share /= args.bootstrap

    rows = []
    for k in range(n + 1):
        row = {"n_turn_away": k, "n_selfish": n - k}
        for name, value in point[k].items():
            row[name] = dict(value)
            values = np.asarray(draws[k][name], dtype=float)
            with np.errstate(invalid="ignore"):
                low, high = np.percentile(values, [2.5, 97.5], method="nearest")
            row[name]["payoff_ci95"] = [float(low), float(high)]
        row.update(point_checks[k])
        row["pure_equilibrium"] = not (
            point_checks[k]["turn_away_gains_by_switching"] or point_checks[k]["selfish_gains_by_switching"]
        )
        row["bootstrap_share_stable"] = float(stable_share[k])
        rows.append(row)

    equilibria = [[row["n_turn_away"], row["n_selfish"]] for row in rows if row["pure_equilibrium"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "paper": "Hahn et al. (2020), Nash Equilibria in Multi-Agent Swarms (ICAART)",
                "checkpoint": str(args.checkpoint),
                "settings": settings.__dict__,
                "episodes_per_split": args.episodes,
                "frames_per_episode": args.frames,
                "paper_exposure": "10 runs x 10 episodes x 100,000 frames per split",
                "payoff": "frames survived per catch (exposure / catches, inf if never caught), pooled over prey of that strategy; mean_completed_spell is the paper-style average of completed spells",
                "equilibrium": "pure-strategy profile (k TurnAway, n-k SELFish) where no single prey gains by switching; bootstrap_share_stable = share of episode-resampled draws in which that holds",
                "respawn": "uniform random (paper: densest spot by KDE)",
                "splits": rows,
                "pure_equilibria_turn_away_selfish": equilibria,
            },
            indent=2,
            default=lambda value: None if value != value else ("inf" if value == np.inf else value),
        )
        + "\n"
    )
    print("pure equilibria (TurnAway, SELFish):", equilibria)


if __name__ == "__main__":
    main()
