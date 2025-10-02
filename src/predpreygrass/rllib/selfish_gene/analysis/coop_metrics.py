import json
from pathlib import Path

import numpy as np


def load_episode(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def _los_clear(walls, a, b):
    # Bresenham between a and b, excluding endpoints
    (x0, y0), (x1, y1) = a, b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if dx >= dy:
        err = dx / 2.0
        while x != x1:
            if (x, y) not in (a, b) and (x, y) in walls:
                return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if (x, y) not in (a, b) and (x, y) in walls:
                return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return True


def compute_assortment_index(episodes, los_aware=False, n_bootstrap=0, seed=0):
    """
    Assortment Index (AI): average same-root_ancestor neighbor fraction vs a shuffled baseline.
    For each step, for each agent, count neighbors within Chebyshev radius R (from episode config
    kin_density_radius). AI = mean( same_root / max(1, total_neighbors) ) - baseline, where baseline is
    computed by shuffling root assignments per step.
    """
    numer = []
    denom = []
    # Use first episode's config for R
    if not episodes:
        return {"ai": 0.0, "n": 0}
    R = int(episodes[0]["config"].get("kin_density_radius", 2))
    walls = set(tuple(w) for w in episodes[0].get("walls", [])) if los_aware else None

    for ep in episodes:
        for step in ep.get("steps", []):
            agents = step["agents"]
            ids = list(agents.keys())
            roots = [agents[i].get("root_ancestor") for i in ids]
            positions = [agents[i]["position"] for i in ids]
            # Build neighbor list per agent
            for idx, aid in enumerate(ids):
                ra = roots[idx]
                ax, ay = positions[idx]
                total = 0
                same = 0
                for jdx, bid in enumerate(ids):
                    if jdx == idx:
                        continue
                    bx, by = positions[jdx]
                    if max(abs(bx - ax), abs(by - ay)) <= R and (not los_aware or _los_clear(walls, (ax, ay), (bx, by))):
                        total += 1
                        if roots[jdx] == ra and ra is not None:
                            same += 1
                if total > 0:
                    numer.append(same)
                    denom.append(total)

    if not denom:
        return {"ai": 0.0, "n": 0}
    observed = np.mean(np.array(numer) / np.array(denom))

    # Baseline by random shuffling roots per step
    rng = np.random.default_rng(0)
    baseline_samples = []
    for ep in episodes:
        for step in ep.get("steps", []):
            agents = step["agents"]
            ids = list(agents.keys())
            roots = np.array([agents[i].get("root_ancestor") for i in ids], dtype=object)
            positions = [agents[i]["position"] for i in ids]
            if len(ids) < 2:
                continue
            for _ in range(10):
                rng.shuffle(roots)
                loc_num = 0
                loc_den = 0
                for idx, aid in enumerate(ids):
                    ra = roots[idx]
                    ax, ay = positions[idx]
                    total = 0
                    same = 0
                    for jdx in range(len(ids)):
                        if jdx == idx:
                            continue
                        bx, by = positions[jdx]
                        if max(abs(bx - ax), abs(by - ay)) <= R and (not los_aware or _los_clear(walls, (ax, ay), (bx, by))):
                            total += 1
                            if roots[jdx] == ra and ra is not None:
                                same += 1
                    loc_num += same
                    loc_den += total
                if loc_den > 0:
                    baseline_samples.append(loc_num / loc_den)
    baseline = float(np.mean(baseline_samples)) if baseline_samples else 0.0
    point = float(observed - baseline)

    # Optional bootstrap CI over steps
    if n_bootstrap and denom:
        rng = np.random.default_rng(seed)
        ratios = np.array(numer) / np.array(denom)
        bs = []
        for _ in range(int(n_bootstrap)):
            idxs = rng.integers(0, len(ratios), size=len(ratios))
            bs.append(float(np.mean(ratios[idxs])) - baseline)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"ai": point, "n": int(len(denom)), "ci95": [float(lo), float(hi)]}
    return {"ai": point, "n": int(len(denom))}


def compute_assortment_index_by_policy(episodes, los_aware=False, n_bootstrap=0, seed=0):
    # Gather all policy groups present in logs
    groups = set()
    for ep in episodes:
        for step in ep.get("steps", []):
            for ad in step.get("agents", {}).values():
                g = ad.get("policy_group")
                if g:
                    groups.add(g)
    results = {}
    for g in sorted(groups):
        numer, denom = [], []
        if not episodes:
            results[g] = {"ai": 0.0, "n": 0}
            continue
        R = int(episodes[0]["config"].get("kin_density_radius", 2))
        walls = set(tuple(w) for w in episodes[0].get("walls", [])) if los_aware else None
        for ep in episodes:
            for step in ep.get("steps", []):
                agents = step["agents"]
                ids = [i for i,a in agents.items() if a.get("policy_group") == g]
                if not ids:
                    continue
                roots = [agents[i].get("root_ancestor") for i in ids]
                positions = [agents[i]["position"] for i in ids]
                for idx, aid in enumerate(ids):
                    ra = roots[idx]
                    ax, ay = positions[idx]
                    total = 0
                    same = 0
                    for jdx, bid in enumerate(ids):
                        if jdx == idx:
                            continue
                        bx, by = positions[jdx]
                        if max(abs(bx - ax), abs(by - ay)) <= R and (not los_aware or _los_clear(walls, (ax, ay), (bx, by))):
                            total += 1
                            if roots[jdx] == ra and ra is not None:
                                same += 1
                    if total > 0:
                        numer.append(same)
                        denom.append(total)
        if not denom:
            results[g] = {"ai": 0.0, "n": 0}
            continue
        observed = float(np.mean(np.array(numer) / np.array(denom)))
        # Baseline by random shuffling roots per step (within group)
        rng = np.random.default_rng(seed)
        baseline_samples = []
        for ep in episodes:
            for step in ep.get("steps", []):
                agents = step["agents"]
                ids = [i for i,a in agents.items() if a.get("policy_group") == g]
                if len(ids) < 2:
                    continue
                roots = np.array([agents[i].get("root_ancestor") for i in ids], dtype=object)
                positions = [agents[i]["position"] for i in ids]
                for _ in range(10):
                    rng.shuffle(roots)
                    loc_num = 0
                    loc_den = 0
                    for idx in range(len(ids)):
                        ra = roots[idx]
                        ax, ay = positions[idx]
                        total = 0
                        same = 0
                        for jdx in range(len(ids)):
                            if jdx == idx:
                                continue
                            bx, by = positions[jdx]
                            if max(abs(bx - ax), abs(by - ay)) <= R and (not los_aware or _los_clear(walls, (ax, ay), (bx, by))):
                                total += 1
                                if roots[jdx] == ra and ra is not None:
                                    same += 1
                        loc_num += same
                        loc_den += total
                    if loc_den > 0:
                        baseline_samples.append(loc_num / loc_den)
        baseline = float(np.mean(baseline_samples)) if baseline_samples else 0.0
        point = float(observed - baseline)
        if n_bootstrap and denom:
            rng = np.random.default_rng(seed)
            ratios = np.array(numer) / np.array(denom)
            bs = []
            for _ in range(int(n_bootstrap)):
                idxs = rng.integers(0, len(ratios), size=len(ratios))
                bs.append(float(np.mean(ratios[idxs])) - baseline)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            results[g] = {"ai": point, "n": int(len(denom)), "ci95": [float(lo), float(hi)]}
        else:
            results[g] = {"ai": point, "n": int(len(denom))}
    return results


def compute_kin_proximity_advantage(episodes, los_aware=False, n_bootstrap=0, seed=0):
    """
    Kin Proximity Advantage (KPA): Compare reproduction rate when at least one same-root neighbor
    is within radius R vs when none are. We approximate by checking if offspring_count increases
    next step while kin present in current step.
    """
    if not episodes:
        return {"kpa": 0.0, "n_with": 0, "n_without": 0}
    R = int(episodes[0]["config"].get("kin_density_radius", 2))
    walls = set(tuple(w) for w in episodes[0].get("walls", [])) if los_aware else None

    with_kin_events = 0
    with_kin_trials = 0
    without_kin_events = 0
    without_kin_trials = 0

    for ep in episodes:
        steps = ep.get("steps", [])
        for t in range(len(steps) - 1):
            cur_agents = steps[t]["agents"]
            nxt_agents = steps[t + 1]["agents"]
            for aid, ad in cur_agents.items():
                if aid not in nxt_agents:
                    continue
                ra = ad.get("root_ancestor")
                ax, ay = ad["position"]
                # kin present?
                kin_present = False
                for bid, bd in cur_agents.items():
                    if bid == aid:
                        continue
                    bx, by = bd["position"]
                    if (
                        max(abs(bx - ax), abs(by - ay)) <= R
                        and bd.get("root_ancestor") == ra
                        and ra is not None
                        and (not los_aware or _los_clear(walls, (ax, ay), (bx, by)))
                    ):
                        kin_present = True
                        break
                # reproduction next step?
                off_now = ad.get("offspring_count", 0)
                off_next = nxt_agents[aid].get("offspring_count", off_now)
                reproduced = off_next > off_now
                if kin_present:
                    with_kin_trials += 1
                    if reproduced:
                        with_kin_events += 1
                else:
                    without_kin_trials += 1
                    if reproduced:
                        without_kin_events += 1

    p_with = with_kin_events / with_kin_trials if with_kin_trials else 0.0
    p_without = without_kin_events / without_kin_trials if without_kin_trials else 0.0
    point = float(p_with - p_without)
    if n_bootstrap and (with_kin_trials + without_kin_trials) > 0:
        rng = np.random.default_rng(seed)
        # Build event/trial arrays for bootstrap
        vec = ([1] * with_kin_events + [0] * (with_kin_trials - with_kin_events),
               [1] * without_kin_events + [0] * (without_kin_trials - without_kin_events))
        # Flatten with labels: 1 for with, 0 for without
        with_vec = np.array(vec[0], dtype=int)
        without_vec = np.array(vec[1], dtype=int)
        n_with = len(with_vec)
        n_without = len(without_vec)
        bs = []
        bs_iters = int(n_bootstrap)
        for _ in range(bs_iters):
            if n_with:
                p_w = float(np.mean(with_vec[rng.integers(0, n_with, size=n_with)]))
            else:
                p_w = 0.0
            if n_without:
                p_wo = float(np.mean(without_vec[rng.integers(0, n_without, size=n_without)]))
            else:
                p_wo = 0.0
            bs.append(p_w - p_wo)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"kpa": point, "n_with": int(with_kin_trials), "n_without": int(without_kin_trials), "ci95": [float(lo), float(hi)]}
    return {"kpa": point, "n_with": int(with_kin_trials), "n_without": int(without_kin_trials)}


def compute_kin_proximity_advantage_by_policy(episodes, los_aware=False, n_bootstrap=0, seed=0):
    groups = set()
    for ep in episodes:
        for step in ep.get("steps", []):
            for ad in step.get("agents", {}).values():
                g = ad.get("policy_group")
                if g:
                    groups.add(g)
    results = {}
    R = int(episodes[0]["config"].get("kin_density_radius", 2)) if episodes else 2
    walls = set(tuple(w) for w in episodes[0].get("walls", [])) if (episodes and los_aware) else None
    for g in sorted(groups):
        with_kin_events = with_kin_trials = 0
        without_kin_events = without_kin_trials = 0
        for ep in episodes:
            steps = ep.get("steps", [])
            for t in range(len(steps) - 1):
                cur_agents = steps[t]["agents"]
                nxt_agents = steps[t + 1]["agents"]
                for aid, ad in cur_agents.items():
                    if ad.get("policy_group") != g:
                        continue
                    if aid not in nxt_agents:
                        continue
                    ra = ad.get("root_ancestor")
                    ax, ay = ad["position"]
                    kin_present = False
                    for bid, bd in cur_agents.items():
                        if bid == aid:
                            continue
                        if bd.get("policy_group") != g:
                            continue
                        bx, by = bd["position"]
                        if (
                            max(abs(bx - ax), abs(by - ay)) <= R
                            and bd.get("root_ancestor") == ra
                            and ra is not None
                            and (not los_aware or _los_clear(walls, (ax, ay), (bx, by)))
                        ):
                            kin_present = True
                            break
                    off_now = ad.get("offspring_count", 0)
                    off_next = nxt_agents[aid].get("offspring_count", off_now)
                    reproduced = off_next > off_now
                    if kin_present:
                        with_kin_trials += 1
                        if reproduced:
                            with_kin_events += 1
                    else:
                        without_kin_trials += 1
                        if reproduced:
                            without_kin_events += 1
        p_with = with_kin_events / with_kin_trials if with_kin_trials else 0.0
        p_without = without_kin_events / without_kin_trials if without_kin_trials else 0.0
        point = float(p_with - p_without)
        if n_bootstrap and (with_kin_trials + without_kin_trials) > 0:
            rng = np.random.default_rng(seed)
            with_vec = np.array([1] * with_kin_events + [0] * (with_kin_trials - with_kin_events), dtype=int)
            without_vec = np.array([1] * without_kin_events + [0] * (without_kin_trials - without_kin_events), dtype=int)
            n_with, n_without = len(with_vec), len(without_vec)
            bs = []
            for _ in range(int(n_bootstrap)):
                p_w = float(np.mean(with_vec[rng.integers(0, n_with, size=n_with)])) if n_with else 0.0
                p_wo = float(np.mean(without_vec[rng.integers(0, n_without, size=n_without)])) if n_without else 0.0
                bs.append(p_w - p_wo)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            results[g] = {"kpa": point, "n_with": int(with_kin_trials), "n_without": int(without_kin_trials), "ci95": [float(lo), float(hi)]}
        else:
            results[g] = {"kpa": point, "n_with": int(with_kin_trials), "n_without": int(without_kin_trials)}
    return results


def main(log_dir, los_aware=False, n_bootstrap=0, seed=0, by_policy=True):
    p = Path(log_dir)
    files = sorted(p.glob("episode_*.json"))
    episodes = [load_episode(fp) for fp in files]
    ai = compute_assortment_index(episodes, los_aware=los_aware, n_bootstrap=n_bootstrap, seed=seed)
    kpa = compute_kin_proximity_advantage(episodes, los_aware=los_aware, n_bootstrap=n_bootstrap, seed=seed)
    out = {"ai": ai, "kpa": kpa}
    if by_policy:
        out["ai_by_policy"] = compute_assortment_index_by_policy(episodes, los_aware=los_aware, n_bootstrap=n_bootstrap, seed=seed)
        out["kpa_by_policy"] = compute_kin_proximity_advantage_by_policy(episodes, los_aware=los_aware, n_bootstrap=n_bootstrap, seed=seed)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute cooperation metrics from coop logs")
    parser.add_argument("--log-dir", default="output/coop_logs", help="Directory containing episode_*.json logs")
    parser.add_argument("--los-aware", action="store_true", help="Use LOS blocking when counting neighbors")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations for CI (0 to disable)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for bootstrap and shuffling")
    parser.add_argument("--by-policy", action="store_true", help="Include per-policy breakdowns in output")
    args = parser.parse_args()
    main(args.log_dir, los_aware=args.los_aware, n_bootstrap=args.bootstrap, seed=args.seed, by_policy=args.by_policy or True)
