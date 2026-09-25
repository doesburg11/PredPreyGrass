"""
Behaviour analysis of trained predator_bands policies (no training): do the sexes specialize, do members stay with their
band, and how does a single woman (or anyone without a partner) get her food? Roll out saved checkpoints in the env each run
was trained in (the config comes from run_config.json), with an instrumented copy of the env that only records.

Reported (conditional on the lives/decisions that occurred in these rollouts; descriptive, not causal. Per-life tables are pooled
over lives; approach and cohesion figures are ratios of totals over decisions, so episodes with more decisions weigh more; the
bootstrap resamples whole episodes):

 A. Energy sources by sex: gross own foraging (prey vs fruit) and everything received from / given to others, split into band
    sharing, parental care and mate gifts, and the fraction of each sex's meat that came from others.
 B. Founders by role (founding couple's male/female, founders' children, unpaired adults; later-born separately): life length,
    own forage, shares received, fraction of meat from sharing. This is the "how does a single woman survive" table.
 C. Approach bias by sex toward prey, fruit, same-band members and other-band members, on every decision where such a target is
    visible: E_uniform[d_after] - E_policy[d_after] (positive = the policy's intended move ends closer than a random mover's would),
    with a 95% episode-cluster percentile bootstrap. Moves onto a cell held by another predator are treated as staying put, as in
    the env; other blocking (prey, edges) is not modelled beyond clipping to the grid.
 D. Cohesion: mean Chebyshev distance to the nearest same-band and other-band predator (only over decisions where such a predator
    exists), and the fraction of ALL decisions with at least one same-band member inside band_share_range, by time bucket. The
    uniform-random baseline (same env and the same seeds, so the same initial layouts) is shown for the first 100 steps only:
    random-policy predators die out quickly, so later buckets would compare against a small surviving subset.

Caveats: rollouts mix action preference with the states the policy creates; lives cut off at the episode end are included in the
per-life means (their counts are shown); the random baseline is descriptive, not a controlled counterfactual; one checkpoint and
one seed per run.

Example:
  python -m predpreygrass.non_evolutionary.predator_bands.analyze_band_behavior \
      --run MEAT060_s42=~/simulation_results/ray_results/PPO_PREDATOR_BANDS_CALIB_F008_MEAT060_SEED42 --checkpoint 19 --episodes 20
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_bands.predpreygrass_rllib_env import PredPreyGrass

POLICY_IDS = ("predator_male_policy", "predator_female_policy")
SEXES = ("predator_male", "predator_female")
TARGETS = ("prey", "fruit", "same_band", "other_band")
BUCKETS = ((0, 100), (100, 300), (300, 10**9))
ROLES = ("couple_male", "couple_female", "child", "single_male", "single_female", "born")


def policy_of(agent_id):
    return "predator_male_policy" if "predator_male" in agent_id else "predator_female_policy"


def load_modules(checkpoint_dir):
    from ray.rllib.core.rl_module.rl_module import RLModule

    root = os.path.join(checkpoint_dir, "learner_group", "learner", "rl_module")
    return {pid: RLModule.from_checkpoint(os.path.join(root, pid)) for pid in POLICY_IDS}


def module_probs(module, observations):
    x = torch.as_tensor(np.stack(observations), dtype=torch.float32)
    with torch.no_grad():
        logits = module._forward_inference({"obs": x})["action_dist_inputs"]
    probs = torch.softmax(logits, dim=-1).double().numpy()
    return probs / probs.sum(axis=1, keepdims=True)


def action_geometry(pos, targets, moves, grid_size, offset, occupied=None):
    """None unless the nearest target is visible (Chebyshev <= offset) and none is co-located; else the destination distance
    to the nearest target for every action. `occupied` (a set of (x, y) cells held by other predators) makes a move onto such a
    cell leave the agent in place, as the env does."""
    if len(targets) == 0:
        return None
    cheb = np.abs(targets - pos).max(axis=1)
    if (cheb == 0).any() or cheb.min() > offset:
        return None
    new = np.clip(pos + moves, 0, grid_size - 1)
    if occupied:
        blocked = np.array([tuple(int(v) for v in c) in occupied and tuple(int(v) for v in c) != tuple(int(v) for v in pos) for c in new])
        new = np.where(blocked[:, None], pos, new)
    delta = new[:, None, :] - targets[None, :, :]
    return np.sqrt((delta ** 2).sum(-1)).min(axis=1)


class InstrumentedBandsEnv(PredPreyGrass):
    """Records per-agent gross own foraging (meat / fruit) and what each kind of transfer moved (band sharing, parental care, mate
    gifts); changes no dynamics."""

    CATEGORIES = ("band", "care", "gift")

    def reset(self, *args, **kwargs):
        self.own = {"meat": defaultdict(float), "fruit": defaultdict(float)}
        self.n_items = {"meat": defaultdict(int), "fruit": defaultdict(int)}  # positive-gain forage events
        self.recv = {c: {"meat": defaultdict(float), "fruit": defaultdict(float)} for c in self.CATEGORIES}
        self.given = {c: {"meat": defaultdict(float), "fruit": defaultdict(float)} for c in self.CATEGORIES}
        return super().reset(*args, **kwargs)

    def _track(self, category, kind, before):
        for a, e0 in before.items():
            d = self.agent_energies[a] - e0
            if d > 1e-12:
                self.recv[category][kind][a] += d
            elif d < -1e-12:
                self.given[category][kind][a] += -d

    def _apply_band_share(self, agent, gained, is_fruit):
        kind = "fruit" if is_fruit else "meat"
        if gained > 0:
            self.own[kind][agent] += gained
            self.n_items[kind][agent] += 1
        before = {a: self.agent_energies[a] for a in self.predator_positions}
        result = super()._apply_band_share(agent, gained, is_fruit)
        self._track("band", kind, before)
        return result

    def _share_energy_with_offspring(self, agent, energy_gained, is_fruit=False):
        before = {a: self.agent_energies[a] for a in self.predator_positions}
        result = super()._share_energy_with_offspring(agent, energy_gained, is_fruit=is_fruit)
        self._track("care", "fruit" if is_fruit else "meat", before)
        return result

    def _apply_male_gift(self, agent, energy_gained):
        before = {a: self.agent_energies[a] for a in self.predator_positions}
        result = super()._apply_male_gift(agent, energy_gained)
        self._track("gift", "meat", before)
        return result

    def _apply_female_gift(self, agent, fruit_gained):
        before = {a: self.agent_energies[a] for a in self.predator_positions}
        result = super()._apply_female_gift(agent, fruit_gained)
        self._track("gift", "fruit", before)
        return result


def founder_roles(env):
    """Roles of the founding predators at reset: couple members, their children, unpaired adults."""
    roles = {}
    for a in env.predator_positions:
        male = "predator_male" in a
        if a in env.agent_parents:
            roles[a] = "child"
        elif a in env.agent_mate:
            roles[a] = "couple_male" if male else "couple_female"
        else:
            roles[a] = "single_male" if male else "single_female"
    return roles


def cohesion_features(env, agent):
    """(distance to nearest same-band predator or inf, distance to nearest other-band predator or inf, same-band count in range,
    same-band positions, other-band positions)"""
    pos = np.array(env.agent_positions[agent])
    band = env.agent_band.get(agent)
    same, other = [], []
    for o, p in env.predator_positions.items():
        if o == agent or o not in env.agent_band:
            continue
        (same if env.agent_band[o] == band else other).append(p)
    same = np.array(same, dtype=int).reshape(-1, 2)
    other = np.array(other, dtype=int).reshape(-1, 2)
    d_same = float(np.abs(same - pos).max(axis=1).min()) if len(same) else float("inf")
    d_other = float(np.abs(other - pos).max(axis=1).min()) if len(other) else float("inf")
    n_in_range = int((np.abs(same - pos).max(axis=1) <= env.band_share_range).sum()) if len(same) else 0
    return d_same, d_other, n_in_range, same, other


def rollout(config, modules, n_episodes, seed0, random_policy=False):
    env = InstrumentedBandsEnv(config)
    moves = np.array([env.action_to_move_tuple[a] for a in range(env.num_actions)])
    offset = (env.predator_obs_range - 1) // 2
    rng = np.random.default_rng(seed0)
    episodes, lives = [], []
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        roles = founder_roles(env)
        active = list(observations.keys())
        first, last = {a: env.current_step for a in active}, {a: env.current_step for a in active}
        approach = {(s, t): [0.0, 0] for s in SEXES for t in TARGETS}
        coh = {(s, b): np.zeros(6) for s in SEXES for b in range(len(BUCKETS))}  # n, sum_dsame, n_dsame, sum_dother, n_dother, n_in_range
        while True:
            targets = {
                "prey": np.array(list(env.prey_positions.values()), dtype=int).reshape(-1, 2),
                "fruit": np.array([p for f, p in env.fruit_positions.items() if env.fruit_energies[f] > 0], dtype=int).reshape(-1, 2),
            }
            bucket = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= env.current_step < hi)
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                if random_policy:
                    rows = np.full((len(agents), env.num_actions), 1.0 / env.num_actions)
                else:
                    rows = module_probs(modules[pid], [observations[a] for a in agents])
                sex = "predator_male" if pid == "predator_male_policy" else "predator_female"
                for i, agent in enumerate(agents):
                    probs = rows[i]
                    pos = np.array(env.agent_positions[agent], dtype=int)
                    d_same, d_other, n_in_range, same, other = cohesion_features(env, agent)
                    c = coh[(sex, bucket)]
                    c[0] += 1
                    if np.isfinite(d_same):
                        c[1] += d_same
                        c[2] += 1
                    if np.isfinite(d_other):
                        c[3] += d_other
                        c[4] += 1
                    c[5] += n_in_range > 0
                    occupied = {tuple(int(v) for v in p) for o, p in env.predator_positions.items() if o != agent}
                    for name, tg in (("prey", targets["prey"]), ("fruit", targets["fruit"]), ("same_band", same), ("other_band", other)):
                        d_after = action_geometry(pos, tg, moves, env.grid_size, offset, occupied)
                        if d_after is not None:
                            approach[(sex, name)][0] += d_after.mean() - float(probs @ d_after)
                            approach[(sex, name)][1] += 1
                    actions[agent] = int(rng.choice(env.num_actions, p=probs))
            observations, _, terminations, truncations, _ = env.step(actions)
            for a in observations:
                first.setdefault(a, env.current_step)
                last[a] = env.current_step
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        alive_at_end = {a for a in observations if not terminations.get(a, False)}
        for a in first:
            lives.append(
                {
                    "episode": ep,
                    "sex": "predator_male" if "predator_male" in a else "predator_female",
                    "role": roles.get(a, "born"),
                    "life": last[a] - first[a] + 1,
                    "own_meat": env.own["meat"][a], "own_fruit": env.own["fruit"][a],
                    "recv_meat": sum(env.recv[c]["meat"][a] for c in env.CATEGORIES),
                    "recv_fruit": sum(env.recv[c]["fruit"][a] for c in env.CATEGORIES),
                    "given_meat": sum(env.given[c]["meat"][a] for c in env.CATEGORIES),
                    "given_fruit": sum(env.given[c]["fruit"][a] for c in env.CATEGORIES),
                    "band_recv_meat": env.recv["band"]["meat"][a], "care_recv_meat": env.recv["care"]["meat"][a],
                    "gift_recv_meat": env.recv["gift"]["meat"][a],
                    "censored": a in alive_at_end,
                }
            )
        episodes.append(
            {"approach": approach, "coh": coh, "length": env.current_step,
             "males": env.current_num_predator_male, "females": env.current_num_predator_female}
        )
    return episodes, lives


# ---- statistics -----------------------------------------------------------------------------------------------------------
def ratio_ci(nums, dens, rng, n_boot=2000):
    """sum(num)/sum(den) over episodes with a 95% episode-cluster percentile bootstrap; None if there is no support."""
    nums, dens = np.asarray(nums, dtype=float), np.asarray(dens, dtype=float)
    if dens.sum() == 0:
        return None
    point = nums.sum() / dens.sum()
    if len(nums) < 5:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, len(nums), size=(n_boot, len(nums)))
    d = dens[idx].sum(1)
    ok = d > 0
    boots = nums[idx].sum(1)[ok] / d[ok]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def fmt(r, digits=3):
    return "n/a" if r is None else f"{r[0]:+.{digits}f} [{r[1]:+.{digits}f},{r[2]:+.{digits}f}]"


def summarize(label, episodes, lives, rng, random_episodes=None):
    out = [f"== {label}: {len(episodes)} episodes, mean length {np.mean([e['length'] for e in episodes]):.0f}, "
           f"final males {np.mean([e['males'] for e in episodes]):.1f}, final females {np.mean([e['females'] for e in episodes]):.1f}"]
    # A. energy sources by sex
    out.append("A. energy sources per life (pooled mean over lives), by sex; 'received' = band sharing + parental care + mate gifts")
    for sex in SEXES:
        rows = [l for l in lives if l["sex"] == sex]
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        own_meat, recv_meat = m("own_meat"), m("recv_meat")
        meat_in = own_meat - m("given_meat") + recv_meat
        share = 100 * m("recv_meat") / (own_meat + m("recv_meat")) if own_meat + m("recv_meat") > 0 else float("nan")
        own_total = own_meat + m("own_fruit")
        out.append(
            f"   {sex[9:]:6} n={len(rows):5d} life={m('life'):6.1f} | own meat {own_meat:6.2f} fruit {m('own_fruit'):6.2f} "
            f"(prey share of own foraging {100 * own_meat / own_total if own_total else float('nan'):4.1f}%) | "
            f"received meat {recv_meat:5.2f} (band {m('band_recv_meat'):4.2f}, care {m('care_recv_meat'):4.2f}, gifts {m('gift_recv_meat'):4.2f}) "
            f"fruit {m('recv_fruit'):5.2f}, gave meat {m('given_meat'):5.2f} fruit {m('given_fruit'):5.2f} | "
            f"meat from others {share:4.1f}% of (own+received) meat | net meat {meat_in:6.2f}"
        )
    # B. by founder role
    out.append("B. founders by role (pooled mean per life)")
    for role in ROLES:
        rows = [l for l in lives if l["role"] == role]
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        denom = m("own_meat") + m("recv_meat")
        out.append(
            f"   {role:14} n={len(rows):5d} life={m('life'):6.1f} censored={sum(r['censored'] for r in rows):4d} | own meat {m('own_meat'):6.2f} "
            f"fruit {m('own_fruit'):6.2f} | received meat {m('recv_meat'):5.2f} fruit {m('recv_fruit'):5.2f} | "
            f"meat from others {100 * m('recv_meat') / denom if denom > 0 else float('nan'):4.1f}% (band {m('band_recv_meat'):4.2f}, care {m('care_recv_meat'):4.2f})"
        )
    # C. approach bias
    out.append("C. approach bias toward each target (positive = closer than a random mover), 95% episode-cluster bootstrap")
    for sex in SEXES:
        parts = []
        for t in TARGETS:
            r = ratio_ci([e["approach"][(sex, t)][0] for e in episodes], [e["approach"][(sex, t)][1] for e in episodes], rng)
            n = sum(e["approach"][(sex, t)][1] for e in episodes)
            parts.append(f"{t}={fmt(r)} (n={n})")
        out.append(f"   {sex[9:]:6} " + " | ".join(parts))
    # D. cohesion
    out.append("D. cohesion: nearest same-band / other-band distance (over decisions where one exists) and share of all decisions with a same-band member within share range; random baseline for steps 0-100 only")
    for sex in SEXES:
        for b, (lo, hi) in enumerate(BUCKETS):
            def col(eps, k):
                return [e["coh"][(sex, b)][k] for e in eps]
            line = f"   {sex[9:]:6} steps {lo}-{hi if hi < 10**8 else 'end'}: "
            n = sum(col(episodes, 0))
            if n == 0:
                out.append(line + "no decisions")
                continue
            near = ratio_ci(col(episodes, 1), col(episodes, 2), rng)
            far = ratio_ci(col(episodes, 3), col(episodes, 4), rng)
            cover = ratio_ci(col(episodes, 5), col(episodes, 0), rng)
            line += f"trained same-band dist {fmt(near, 2)}, other-band {fmt(far, 2)}, within-range {fmt(cover, 2)} (n={int(n)})"
            if random_episodes is not None and b == 0 and sum(col(random_episodes, 0)) > 0:
                rn = ratio_ci(col(random_episodes, 1), col(random_episodes, 2), rng)
                rc = ratio_ci(col(random_episodes, 5), col(random_episodes, 0), rng)
                line += f" | random same-band dist {fmt(rn, 2)}, within-range {fmt(rc, 2)} (n={int(sum(col(random_episodes, 0)))})"
            out.append(line)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=19)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--random-episodes", type=int, default=40, help="uniform-random baseline episodes for the cohesion comparison")
    parser.add_argument("--seed", type=int, default=6000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    for spec in args.run:
        label, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
        modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
        episodes, lives = rollout(config, modules, args.episodes, args.seed)
        random_eps, _ = rollout(config, None, args.random_episodes, args.seed, random_policy=True)
        for line in summarize(f"{label} ckpt{args.checkpoint}", episodes, lives, rng, random_eps):
            print(line, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
