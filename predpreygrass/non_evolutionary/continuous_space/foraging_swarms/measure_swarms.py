"""Measure swarming, and the use of social information, in trained policies
(README, "Stage 0" and "Measurements").

Plays seeded episodes of one checkpoint with sampled actions (as in
training), in parallel, and records for each species:

- spacing: nearest-neighbour distance and the Clark-Evans ratio R against
  random placement of the same number of animals;
- alignment: local alignment with neighbours within ALIGN_RADIUS against
  the same positions with shuffled headings, and global polarization;
  for moving animals at least MIN_AGE decisions old only, split by
  whether the animal's target (grass for prey, prey for predators) is in
  view;
- groups: share of animals in groups of at least 3 (single linkage at
  LINKS), against random placement; how long pairs stay together (parent
  and offspring left out); group lifetimes;
- whether grouping pays: catches and grass per prey-decision, in a group
  or not;
- target out of view: how often an animal sees no target, and how often it
  then sees a conspecific;
- losing the target (Hahn et al. 2020, Fig. 4): distance to the nearest
  target over the 20 decisions after it leaves the view cone, split by
  whether a conspecific was in view at that moment;
- the counterfactual probe: on observations without the target in view and
  with a moving conspecific in view, rotate the conspecifics' velocities
  and measure how much the policy's direction distribution changes, and
  whether its mean direction turns with them.

Every number is also written to a JSON file. Usage, from the repository
root:

    python -m predpreygrass.non_evolutionary.continuous_space.foraging_swarms.measure_swarms \\
        <checkpoint> <label> [--episodes 50] [--seed 0] [--workers 25] \\
        [--override '{"prey_observe_neighbor_movement": false}']

--override changes the environment at test time (the policy is the
checkpoint's): a test-time ablation, not a trained condition.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import swarm_metrics as sm
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.checkpoints import (
    SPECIES,
    checkpoint_env_config,
    load_modules,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import (
    DIRECTIONS,
    SELF_SIZE,
    SLOT_SIZE,
    SPEEDS,
    PredPreyGrassContinuous,
)
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.tournament import classify

HERE = Path(__file__).resolve().parent
MIN_AGE = 10  # decisions: newborns start with a fixed velocity next to their parent
MOVING_SPEED = 0.5  # arena units per physics step; below counts as standing
ALIGN_RADIUS = 96.0  # three body diameters
LINKS = (64.0, 96.0)  # two and three body diameters
GROUP_LINK = 96.0  # for "in a group", pair and group persistence
MIN_GROUP = 3
LOSS_WINDOW = 20  # decisions followed after losing the target
PROBE_ANGLES = (45, 90, 135, 180, 225, 270, 315)
PROBE_EVERY = 4  # keep every 4th qualifying observation for the probe
PROBE_CAP = 1500  # per species and episode
# The probe's turn is only defined when the direction distribution has a
# clear mean direction: both before and after the rotation, the length of
# its mean unit vector must be at least this.
PROBE_CONCENTRATED = 0.5
BOOTSTRAP = 2000  # resamples of episodes for the intervals


def layout(env):
    """Index of the first predator slot, first prey slot and the grass
    flag in an observation."""
    n_pred, n_prey = env.observe_count["predator"], env.observe_count["prey"]
    return {
        "predator": SELF_SIZE,
        "prey": SELF_SIZE + SLOT_SIZE * n_pred,
        "grass": SELF_SIZE + SLOT_SIZE * (n_pred + n_prey),
        "count": {"predator": n_pred, "prey": n_prey},
    }


def target_in_view(obs, species, lay):
    """Prey look for grass, predators for prey."""
    return obs[lay["grass"]] > 0 if species == "prey" else obs[lay["prey"]] > 0


def conspecific_in_view(obs, species, lay):
    return lay["count"][species] > 0 and obs[lay[species]] > 0


def moving_conspecific_in_view(obs, species, lay):
    start = lay[species]
    for k in range(lay["count"][species]):
        slot = obs[start + SLOT_SIZE * k : start + SLOT_SIZE * (k + 1)]
        if slot[0] > 0 and math.hypot(slot[4], slot[5]) > 0:
            return True
    return False


_NULL_GROUP: dict = {}


def random_group_share(n, width, height, link, samples=200, seed=0):
    """Share of n uniformly placed animals in groups of at least MIN_GROUP
    at this link distance (Monte Carlo, cached)."""
    key = (n, width, height, link, samples, seed)
    if key not in _NULL_GROUP:
        rng = np.random.default_rng(seed)
        shares = []
        for _ in range(samples):
            positions = rng.uniform((0, 0), (width, height), size=(n, 2))
            shares.append(float((sm.group_sizes(sm.clusters(positions, width, height, link)) >= MIN_GROUP).mean()))
        _NULL_GROUP[key] = float(np.mean(shares))
    return _NULL_GROUP[key]


def nearest_distance(position, targets, width, height):
    if len(targets) == 0:
        return math.nan
    d = targets - position
    d[:, 0] -= width * np.round(d[:, 0] / width)
    d[:, 1] -= height * np.round(d[:, 1] / height)
    return float(np.hypot(d[:, 0], d[:, 1]).min())


def sample_actions(modules, obs, generator):
    """As in training: sampled, one batch per species, agents sorted."""
    from ray.rllib.core.columns import Columns

    actions = {}
    for species in SPECIES:
        agents = sorted(agent for agent in obs if agent.startswith(species))
        if not agents:
            continue
        batch = torch.from_numpy(np.stack([obs[agent] for agent in agents]).astype(np.float32))
        with torch.inference_mode():
            logits = modules[species].forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        chosen = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(1)
        actions.update(zip(agents, chosen.tolist()))
    return actions


class Accumulator:
    """Sums and counts per named quantity, mergeable across episodes."""

    def __init__(self):
        self.sum = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, name, value, count=1):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self.sum[name] += float(value)
        self.n[name] += count

    def add_many(self, name, values):
        values = np.asarray(values, dtype=float)
        if len(values):
            self.sum[name] += float(values.sum())
            self.n[name] += len(values)


def play(env, modules, seed, rng):
    """One episode; returns its raw measurements."""
    generator = torch.Generator().manual_seed(seed)
    width, height = env.width, env.height
    obs, _ = env.reset(seed=seed)
    lay = layout(env)
    acc = Accumulator()
    born_at: dict[str, int] = {}
    parent_of: dict[str, str] = {}
    pairs = {s: sm.PairTracker(lambda a, b: parent_of.get(a) == b or parent_of.get(b) == a) for s in SPECIES}
    groups = {s: sm.GroupTracker(MIN_GROUP) for s in SPECIES}
    had_target: dict[str, bool] = {}
    losses = {s: [] for s in SPECIES}  # finished loss events
    open_losses: dict[str, list] = {}  # agent -> its events still being followed
    probe = {s: [] for s in SPECIES}
    exposure = {"predator": 0.0, "prey": 0.0}
    caught_total = 0
    decision = 0
    while True:
        animals = {"predator": list(env.predators), "prey": list(env.prey)}
        for species in SPECIES:
            for a in animals[species]:
                born_at.setdefault(a.id, decision)
        grass = np.array([(p.position.x, p.position.y) for p in env.grass.patches if env.grass.available(p)]).reshape(-1, 2)
        prey_xy = np.array([(a.position.x, a.position.y) for a in animals["prey"]]).reshape(-1, 2)
        in_group = {}
        for species in SPECIES:
            group = animals[species]
            n = len(group)
            ids = [a.id for a in group]
            positions = np.array([(a.position.x, a.position.y) for a in group], dtype=float).reshape(-1, 2)
            velocities = np.array([(a.velocity.x, a.velocity.y) for a in group], dtype=float).reshape(-1, 2)
            ages = np.array([decision - born_at[i] for i in ids])
            old = ages >= MIN_AGE
            speeds = np.hypot(velocities[:, 0], velocities[:, 1]) if n else np.empty(0)
            moving = old & (speeds > MOVING_SPEED)
            acc.add(f"{species}/moving_share", float(moving.sum()), int(old.sum()))
            # Spacing among animals old enough (newborns sit next to their parent).
            settled = positions[old]
            if len(settled) >= 2:
                nn = sm.nearest_neighbour_distances(settled, width, height)
                acc.add_many(f"{species}/nn_distance", nn)
                acc.add(f"{species}/clark_evans_R", nn.mean() / sm.random_nearest_neighbour_mean(len(settled), width, height))
                acc.add(f"{species}/overlap_share", float((nn < 2 * group[0].radius).mean()))
            if n >= 2:
                acc.add(f"{species}/clark_evans_R_all_ages", sm.nearest_neighbour_distances(positions, width, height).mean()
                        / sm.random_nearest_neighbour_mean(n, width, height))
            # Alignment, split by whether the target is in view.
            if moving.sum() >= 2:
                sees = np.array([target_in_view(obs[i], species, lay) if i in obs else False for i in ids])
                aligned = sm.local_alignment(positions, velocities, width, height, ALIGN_RADIUS, moving)
                shuffled = sm.shuffled_alignment(positions, velocities, width, height, ALIGN_RADIUS, moving, rng, shuffles=5)
                acc.add_many(f"{species}/alignment", aligned)
                acc.add_many(f"{species}/alignment_shuffled", shuffled)
                for label, focal in (("target_in_view", moving & sees), ("target_out_of_view", moving & ~sees)):
                    # Focal animals by target visibility; neighbours any moving animal.
                    acc.add_many(
                        f"{species}/alignment_{label}",
                        sm.local_alignment(positions, velocities, width, height, ALIGN_RADIUS, moving, focal),
                    )
                acc.add(f"{species}/polarization", sm.polarization(velocities, moving))
            # Groups.
            if n:
                for link in LINKS:
                    sizes = sm.group_sizes(sm.clusters(positions[old], width, height, link)) if old.any() else np.empty(0)
                    if len(sizes) >= 2:
                        acc.add(f"{species}/group_share_{int(link)}", float((sizes >= MIN_GROUP).mean()))
                        acc.add(f"{species}/group_share_{int(link)}_null", random_group_share(len(sizes), width, height, link))
                        acc.add(f"{species}/mean_group_size_{int(link)}", float(sizes.mean()))
                        acc.add(f"{species}/largest_group_share_{int(link)}", float(sizes.max() / len(sizes)))
                old_ids = [i for i, o in zip(ids, old) if o]
                pairs[species].update(old_ids, positions[old], width, height, GROUP_LINK)
                labels = sm.clusters(positions[old], width, height, GROUP_LINK) if old.any() else np.empty(0)
                groups[species].update(old_ids, labels)
                sizes = sm.group_sizes(labels) if len(labels) else np.empty(0)
                in_group.update({i: s >= MIN_GROUP for i, s in zip(old_ids, sizes)})
            # Target out of view, and losing it.
            for a, age in zip(group, ages):
                o = obs.get(a.id)
                if o is None or age < MIN_AGE:
                    had_target.pop(a.id, None)
                    continue
                sees = bool(target_in_view(o, species, lay))
                acc.add(f"{species}/target_out_of_view", float(not sees))
                if not sees:
                    acc.add(f"{species}/conspecific_in_view_when_target_out", float(conspecific_in_view(o, species, lay)))
                    if moving_conspecific_in_view(o, species, lay) and len(probe[species]) < PROBE_CAP and rng.random() < 1 / PROBE_EVERY:
                        probe[species].append(o.copy())
                targets = grass if species == "prey" else prey_xy
                distance = nearest_distance(np.array([a.position.x, a.position.y]), targets, width, height)
                # Every loss is followed for LOSS_WINDOW decisions, also when
                # the animal loses the target again meanwhile (overlapping
                # windows, as in Hahn et al. 2020: regardless of whether the
                # target comes back).
                still_open = []
                for event in open_losses.get(a.id, []):
                    event["distances"].append(distance)
                    if sees and event["reacquired"] is None:
                        event["reacquired"] = len(event["distances"]) - 1
                    if len(event["distances"]) > LOSS_WINDOW:
                        losses[species].append(event)
                    else:
                        still_open.append(event)
                if had_target.get(a.id) and not sees:
                    still_open.append(
                        {
                            "with_conspecific": bool(conspecific_in_view(o, species, lay)),
                            "distances": [distance],
                            "reacquired": None,
                        }
                    )
                if still_open:
                    open_losses[a.id] = still_open
                else:
                    open_losses.pop(a.id, None)
                had_target[a.id] = sees
        # Act.
        actions = sample_actions(modules, obs, generator)
        before = (env.time_step, len(env.predators), len(env.prey))
        prey_before = [a.id for a in env.prey]
        obs, _, terms, truncs, infos = env.step(actions)
        decision += 1
        steps = env.time_step - before[0]
        exposure["predator"] += steps * (before[1] + len(env.predators)) / 2
        exposure["prey"] += steps * (before[2] + len(env.prey)) / 2
        for agent, info in infos.items():
            if info.get("born"):
                parent_of[agent] = info["parent"]
        for agent in prey_before:
            caught = bool(terms.get(agent)) and not infos.get(agent, {}).get("starved")
            caught_total += caught
            if agent in in_group:
                key = "in_group" if in_group[agent] else "alone"
                acc.add(f"prey/caught_{key}", float(caught))
                acc.add(f"prey/grass_{key}", float(infos.get(agent, {}).get("grass_eaten", 0)))
        for agent in list(open_losses):
            if terms.get(agent) or truncs.get(agent):
                species = "predator" if agent.startswith("predator") else "prey"
                losses[species] += open_losses.pop(agent)  # died: shorter records
        if terms["__all__"] or truncs["__all__"]:
            break
    for species in SPECIES:
        losses[species] += [event for agent, events in open_losses.items() if agent.startswith(species) for event in events]
    persistence = {s: pairs[s].close() for s in SPECIES}
    lifetimes = {s: groups[s].close() for s in SPECIES}
    return {
        "seed": seed,
        "outcome": classify(len(env.predators), len(env.prey)),
        "physics_steps": env.time_step,
        "decisions": decision,
        "prey_caught": caught_total,
        "prey_exposure": exposure["prey"],
        "mean_predators": exposure["predator"] / max(env.time_step, 1),
        "mean_prey": exposure["prey"] / max(env.time_step, 1),
        "sums": dict(acc.sum),
        "counts": dict(acc.n),
        "pair_spells": {s: persistence[s] for s in SPECIES},
        "group_lifetimes": {s: lifetimes[s] for s in SPECIES},
        "losses": losses,
        "probe": {s: np.asarray(probe[s], dtype=np.float32) for s in SPECIES},
    }


# --- The counterfactual probe ------------------------------------------------


def direction_distribution(module, batch):
    """Probability of each of the 16 directions (moving actions only,
    renormalized) and of standing still, for a batch of observations."""
    from ray.rllib.core.columns import Columns

    with torch.inference_mode():
        logits = module.forward_inference({Columns.OBS: torch.from_numpy(batch)})[Columns.ACTION_DIST_INPUTS]
    p = torch.softmax(logits, dim=-1).numpy().reshape(len(batch), len(SPEEDS), DIRECTIONS)
    stop = SPEEDS.index(0.0)
    moving = np.delete(p, stop, axis=1).sum(axis=1)
    with np.errstate(invalid="ignore"):
        moving_dist = moving / moving.sum(axis=1, keepdims=True)
    return moving_dist, p[:, stop, :].sum(axis=1)


_ANGLES = np.radians(np.arange(DIRECTIONS) * 360 / DIRECTIONS)
_UNITS = np.stack([np.sin(_ANGLES), -np.cos(_ANGLES)], axis=1)  # direction_vector


def rotate_conspecific_velocities(batch, species, lay, degrees):
    out = batch.copy()
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    start = lay[species]
    for k in range(lay["count"][species]):
        vx, vy = start + SLOT_SIZE * k + 4, start + SLOT_SIZE * k + 5
        x, y = batch[:, vx].copy(), batch[:, vy].copy()
        out[:, vx], out[:, vy] = c * x - s * y, s * x + c * y
    return out


def run_probe(module, batch, species, lay):
    """Per angle: mean total-variation distance between the direction
    distributions with rotated and with actual conspecific velocities (0:
    the policy ignores them), the change in standing probability, and the
    rotation of the policy's mean direction (circular mean and resultant
    length of the turn; following the neighbours fully turns it by the
    angle itself). The turn uses only observations whose direction
    distribution has a clear mean direction both before and after the
    rotation (PROBE_CONCENTRATED); concentrated_share says how many."""
    base, base_stop = direction_distribution(module, batch)
    base_mean = base @ _UNITS
    out = {}
    for degrees in PROBE_ANGLES:
        dist, stop = direction_distribution(module, rotate_conspecific_velocities(batch, species, lay, degrees))
        mean = dist @ _UNITS
        clear = (np.hypot(*base_mean.T) >= PROBE_CONCENTRATED) & (np.hypot(*mean.T) >= PROBE_CONCENTRATED)
        turn = np.arctan2(mean[clear, 1], mean[clear, 0]) - np.arctan2(base_mean[clear, 1], base_mean[clear, 0])
        resultant = np.array([np.cos(turn).mean(), np.sin(turn).mean()]) if clear.any() else np.array([math.nan, math.nan])
        out[str(degrees)] = {
            "tv_distance": float(0.5 * np.abs(dist - base).sum(axis=1).mean()),
            "stop_change": float((stop - base_stop).mean()),
            "mean_turn_degrees": float(math.degrees(math.atan2(resultant[1], resultant[0]))),
            "turn_resultant": float(math.hypot(*resultant)),
            "concentrated_share": float(clear.mean()),
        }
    # Scale for the TV distances: the same policy with the conspecifics'
    # velocities zeroed (what condition B sees), and with its own velocity
    # turned by 90 degrees instead.
    zeroed = batch.copy()
    own = batch.copy()
    for k in range(lay["count"][species]):
        v = lay[species] + SLOT_SIZE * k + 4
        zeroed[:, v : v + 2] = 0.0
    own[:, 0], own[:, 1] = -batch[:, 1], batch[:, 0]
    for name, changed in (("zeroed_conspecific_velocities", zeroed), ("own_velocity_turned_90", own)):
        dist, _ = direction_distribution(module, changed)
        out[name] = {"tv_distance": float(0.5 * np.abs(dist - base).sum(axis=1).mean())}
    return out


# --- Running and summarizing ---------------------------------------------------

_worker: dict = {}


def _init(checkpoint, env_config):
    torch.set_num_threads(1)
    _worker["env"] = PredPreyGrassContinuous(env_config)
    _worker["modules"] = load_modules(checkpoint)


def _episode(seed):
    return play(_worker["env"], _worker["modules"], seed, np.random.default_rng(10_000 + seed))


# Quantities with an interval: (name, key, key of the null or comparison
# subtracted from it, or None).
INTERVALS = [
    (f"{sp}/{name}", a, b)
    for sp in SPECIES
    for name, a, b in (
        ("clark_evans_R", f"{sp}/clark_evans_R", None),
        ("alignment_minus_shuffled", f"{sp}/alignment", f"{sp}/alignment_shuffled"),
        ("alignment_target_out_of_view_minus_shuffled", f"{sp}/alignment_target_out_of_view", f"{sp}/alignment_shuffled"),
        ("group_share_96_minus_random", f"{sp}/group_share_96", f"{sp}/group_share_96_null"),
    )
] + [("prey/caught_in_group_minus_alone", "prey/caught_in_group", "prey/caught_alone")]


def episode_intervals(episodes, rng, resamples=BOOTSTRAP):
    """95% percentile intervals for INTERVALS, resampling whole episodes
    (decisions within an episode are not independent). They describe these
    policies over episodes, not the variation between training runs."""
    out = {}
    for name, a, b in INTERVALS:
        keys = [a] if b is None else [a, b]
        sums = np.array([[e["sums"].get(k, 0.0) for k in keys] for e in episodes])
        counts = np.array([[e["counts"].get(k, 0) for k in keys] for e in episodes])
        if (counts.sum(axis=0) == 0).any():
            continue

        def statistic(index):
            means = sums[index].sum(axis=0) / np.maximum(counts[index].sum(axis=0), 1)
            return means[0] if b is None else means[0] - means[1]

        draws = [statistic(rng.integers(len(episodes), size=len(episodes))) for _ in range(resamples)]
        out[name] = {
            "estimate": float(statistic(np.arange(len(episodes)))),
            "low": float(np.percentile(draws, 2.5)),
            "high": float(np.percentile(draws, 97.5)),
        }
    return out


def summarize(episodes, modules, lay):
    sums, counts = defaultdict(float), defaultdict(int)
    for e in episodes:
        for k, v in e["sums"].items():
            sums[k] += v
        for k, v in e["counts"].items():
            counts[k] += v
    means = {k: sums[k] / counts[k] for k in sorted(sums) if counts[k]}
    outcomes = defaultdict(int)
    for e in episodes:
        outcomes[e["outcome"]] += 1
    summary = {
        "episodes": len(episodes),
        "outcomes": dict(outcomes),
        "coexisting": outcomes["time_limit"],
        "catch_risk_per_prey_per_1000_steps": 1000 * sum(e["prey_caught"] for e in episodes) / max(sum(e["prey_exposure"] for e in episodes), 1e-9),
        "mean_predators": float(np.mean([e["mean_predators"] for e in episodes])),
        "mean_prey": float(np.mean([e["mean_prey"] for e in episodes])),
        "means": means,
        "counts": dict(counts),
        "intervals": episode_intervals(episodes, np.random.default_rng(0)),
    }
    for species in SPECIES:
        finished = [x for e in episodes for x in e["pair_spells"][species][0]]
        censored = [x for e in episodes for x in e["pair_spells"][species][1]]
        g_finished = [x for e in episodes for x in e["group_lifetimes"][species][0]]
        g_censored = [x for e in episodes for x in e["group_lifetimes"][species][1]]
        # Kaplan-Meier: spells cut short by a death or the episode's end
        # count as lasting at least what was seen.
        summary[f"{species}/pair_spells"] = {
            "n": len(finished),
            "censored": len(censored),
            "km_median": sm.km_median(finished, censored),
            "km_share_at_least_10": sm.km_share_lasting(finished, censored, 10),
        }
        summary[f"{species}/group_lifetimes"] = {
            "n": len(g_finished),
            "censored": len(g_censored),
            "km_median": sm.km_median(g_finished, g_censored),
            "km_share_at_least_10": sm.km_share_lasting(g_finished, g_censored, 10),
        }
        loss = {}
        for label, flag in (("with_conspecific", True), ("without_conspecific", False)):
            events = [ev for e in episodes for ev in e["losses"][species] if ev["with_conspecific"] == flag]
            curve = []
            for t in range(LOSS_WINDOW + 1):
                values = [ev["distances"][t] for ev in events if len(ev["distances"]) > t and not math.isnan(ev["distances"][t])]
                curve.append(float(np.mean(values)) if values else math.nan)
            full = [ev for ev in events if len(ev["distances"]) > LOSS_WINDOW]
            loss[label] = {
                "events": len(events),
                "followed_full_window": len(full),
                "mean_distance_by_decision": curve,
                "reacquired_within_window": float(np.mean([ev["reacquired"] is not None for ev in full])) if full else math.nan,
            }
        summary[f"{species}/target_loss"] = loss
        batch = np.concatenate([e["probe"][species] for e in episodes if len(e["probe"][species])] or [np.empty((0, 0), np.float32)])
        summary[f"{species}/probe_observations"] = int(len(batch))
        if len(batch):
            summary[f"{species}/probe"] = run_probe(modules[species], batch, species, lay)
    return summary


def format_summary(label, s):
    m = s["means"]
    lines = [f"== {label}: {s['episodes']} episodes, {s['coexisting']} coexisting {s['outcomes']}",
             f"   catch risk {s['catch_risk_per_prey_per_1000_steps']:.2f} per prey per 1,000 steps; "
             f"predators/prey alive {s['mean_predators']:.1f}/{s['mean_prey']:.1f}"]

    def g(key, fmt="{:.3f}"):
        value = m.get(key)
        return "-" if value is None else fmt.format(value)

    for sp in SPECIES:
        lines.append(f"   {sp}:")
        lines.append(f"     moving {g(sp + '/moving_share')}  NN {g(sp + '/nn_distance', '{:.0f}')}  "
                     f"R {g(sp + '/clark_evans_R')} (all ages {g(sp + '/clark_evans_R_all_ages')})  "
                     f"overlapping {g(sp + '/overlap_share')}")
        lines.append(f"     alignment {g(sp + '/alignment')} vs shuffled {g(sp + '/alignment_shuffled')}; "
                     f"target in view {g(sp + '/alignment_target_in_view')}, out {g(sp + '/alignment_target_out_of_view')}; "
                     f"polarization {g(sp + '/polarization')}")
        for link in LINKS:
            lines.append(f"     groups>={MIN_GROUP} at {int(link)}: {g(f'{sp}/group_share_{int(link)}')} vs random "
                         f"{g(f'{sp}/group_share_{int(link)}_null')}; mean size {g(f'{sp}/mean_group_size_{int(link)}', '{:.2f}')}; "
                         f"largest {g(f'{sp}/largest_group_share_{int(link)}')}")
        ps, gl = s[f"{sp}/pair_spells"], s[f"{sp}/group_lifetimes"]
        lines.append(f"     pair spells (Kaplan-Meier) median {ps['km_median']} (n {ps['n']} + {ps['censored']} censored, "
                     f">=10: {ps['km_share_at_least_10']:.3f}); group lifetimes median {gl['km_median']} "
                     f"(n {gl['n']} + {gl['censored']} censored, >=10: {gl['km_share_at_least_10']:.3f})")
        lines.append(f"     target out of view {g(sp + '/target_out_of_view')}; then conspecific in view "
                     f"{g(sp + '/conspecific_in_view_when_target_out')}")
        for label, d in s[f"{sp}/target_loss"].items():
            c = d["mean_distance_by_decision"]
            lines.append(f"     lost target {label}: {d['events']} events; distance t0 {c[0]:.0f} t5 {c[5]:.0f} "
                         f"t10 {c[10]:.0f} t20 {c[20]:.0f}; back in view within {LOSS_WINDOW}: {d['reacquired_within_window']:.3f}")
        probe = s.get(f"{sp}/probe")
        if probe:
            lines.append(f"     probe ({s[f'{sp}/probe_observations']} obs): " + "; ".join(
                f"{a}deg TV {p['tv_distance']:.3f} turn {p['mean_turn_degrees']:.0f} (R {p['turn_resultant']:.2f}, "
                f"clear {p['concentrated_share']:.2f})"
                for a, p in probe.items() if a in ("90", "180", "270"))
                + f"; references: zeroed TV {probe['zeroed_conspecific_velocities']['tv_distance']:.3f}, "
                f"own velocity 90deg TV {probe['own_velocity_turned_90']['tv_distance']:.3f}")
    if "prey/caught_in_group" in m:
        lines.append(f"   prey caught per decision: in group {m['prey/caught_in_group']:.4f} "
                     f"(n {s['counts']['prey/caught_in_group']}), alone {g('prey/caught_alone', '{:.4f}')}; "
                     f"grass per decision: in group {g('prey/grass_in_group', '{:.4f}')}, alone {g('prey/grass_alone', '{:.4f}')}")
    lines.append("   95% intervals over episodes:")
    for name, i in s["intervals"].items():
        lines.append(f"     {name}: {i['estimate']:.4f} ({i['low']:.4f} to {i['high']:.4f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("label")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(25, os.cpu_count() or 1))
    parser.add_argument("--override", type=json.loads, default={})
    parser.add_argument("--out-dir", default=str(HERE / "analysis" / "results"))
    args = parser.parse_args()
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    env_config = checkpoint_env_config(checkpoint)
    env_config.update(args.override)
    env = PredPreyGrassContinuous(env_config)
    lay = layout(env)
    start = time.time()
    seeds = list(range(args.seed, args.seed + args.episodes))
    with get_context("spawn").Pool(min(args.workers, len(seeds)), initializer=_init, initargs=(checkpoint, env_config)) as pool:
        episodes = pool.map(_episode, seeds, chunksize=1)
    modules = load_modules(checkpoint)
    summary = summarize(episodes, modules, lay)
    summary.update(label=args.label, checkpoint=checkpoint, override=args.override, seeds=seeds,
                   settings=dict(MIN_AGE=MIN_AGE, MOVING_SPEED=MOVING_SPEED, ALIGN_RADIUS=ALIGN_RADIUS, LINKS=LINKS,
                                 GROUP_LINK=GROUP_LINK, MIN_GROUP=MIN_GROUP, LOSS_WINDOW=LOSS_WINDOW),
                   per_episode=[{k: e[k] for k in ("seed", "outcome", "physics_steps", "decisions", "prey_caught",
                                                   "prey_exposure", "mean_predators", "mean_prey")} for e in episodes],
                   wall_seconds=round(time.time() - start, 1))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    text = format_summary(args.label, summary)
    (out / f"{args.label}.txt").write_text(text + "\n")
    print(text)
    print(f"({summary['wall_seconds']} s; written to {out / args.label}.json)")


if __name__ == "__main__":
    sys.exit(main())
