"""
Measure, per predator lifetime, how much energy each sex gets from fruit and from prey.

Rolls out saved checkpoints (stochastic actions, same as analyze_prey_approach_from_checkpoint) on a
thin subclass of the environment that only adds bookkeeping; the environment file is unchanged.

  - Prey energy: the prey's full energy at the moment of a successful catch (returned by
    _resolve_hunting_attempt). Gross: before any mate gift or parental sharing.
  - Fruit energy: the fruit's energy at the moment it is eaten. Gross, same convention.
  - Received / given away: mate gifts (male -> his recorded mate) and parental care (parent ->
    nearby offspring) are measured from the change in every predator's energy around
    _apply_male_gift and _share_energy_with_offspring, so they are exact whatever the rules.
    "Net" intake = own foraging + received - given away.
  - Lifetime: number of steps between the first and the last step the agent appears in the
    observations. Agents still alive at the end of the episode are right-censored (marked).

Example:
  python -m predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_energy_sources \
      --run FORAGING=~/simulation_results/ray_results/PPO_PREDATOR_SEXUAL_REPRODUCTION_FORAGING_CHECK_SEED42 \
      --checkpoint 29 --episodes 20
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from predpreygrass.non_evolutionary.predator_sexual_reproduction.analyze_prey_approach_from_checkpoint import (
    POLICY_IDS,
    load_modules,
    module_probs,
    policy_of,
)
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass

SEXES = ("predator_male", "predator_female")


class InstrumentedEnv(PredPreyGrass):
    """Adds per-agent gross energy-by-source bookkeeping; does not change any dynamics."""

    def _init_energy_log(self):
        self.energy_from_prey = defaultdict(float)
        self.energy_from_fruit = defaultdict(float)
        self.n_prey_caught = defaultdict(int)
        self.n_fruit_eaten = defaultdict(int)
        self._pending_prey = {}
        self.gift_received = defaultdict(float)  # from a mate
        self.gift_given = defaultdict(float)  # to a mate
        self.care_received = defaultdict(float)  # from a parent
        self.care_given = defaultdict(float)  # to offspring

    def _predator_energies(self):
        return {a: e for a, e in self.agent_energies.items() if "predator" in a}

    def _track_transfer(self, before, received, given):
        for a, e0 in before.items():
            e1 = self.agent_energies.get(a)
            if e1 is None:
                continue
            delta = e1 - e0
            if delta > 1e-12:
                received[a] += delta
            elif delta < -1e-12:
                given[a] += -delta

    def reset(self, *args, **kwargs):
        self._init_energy_log()
        return super().reset(*args, **kwargs)

    def _resolve_hunting_attempt(self, agent, *args, **kwargs):
        outcome, reward_delta, energy_gained = super()._resolve_hunting_attempt(agent, *args, **kwargs)
        if outcome == "success":
            self._pending_prey[agent] = energy_gained
            self.energy_from_prey[agent] += energy_gained
            self.n_prey_caught[agent] += 1
        return outcome, reward_delta, energy_gained

    def _apply_male_gift(self, agent, energy_gained):
        before = self._predator_energies()
        result = super()._apply_male_gift(agent, energy_gained)
        self._track_transfer(before, self.gift_received, self.gift_given)
        return result

    def _share_energy_with_offspring(self, agent, energy_gained):
        # Called right after a successful hunt (prey energy) and right after eating a fruit.
        pending = self._pending_prey.get(agent)
        if pending is not None and np.isclose(pending, energy_gained):
            del self._pending_prey[agent]
        elif energy_gained > 0:
            self.energy_from_fruit[agent] += energy_gained
            self.n_fruit_eaten[agent] += 1
        before = self._predator_energies()
        result = super()._share_energy_with_offspring(agent, energy_gained)
        self._track_transfer(before, self.care_received, self.care_given)
        return result


def run_episodes(env_config, modules, n_episodes, seed0, random_policy=False):
    env = InstrumentedEnv(env_config)
    rng = np.random.default_rng(seed0)
    records = []  # one dict per predator agent life
    for ep in range(n_episodes):
        observations, _ = env.reset(seed=seed0 + ep)
        active = list(observations.keys())
        first_seen, last_seen = {}, {}
        for a in active:
            first_seen[a] = last_seen[a] = env.current_step
        while True:
            actions = {}
            for pid in POLICY_IDS:
                agents = [a for a in active if policy_of(a) == pid]
                if not agents:
                    continue
                if random_policy:
                    rows = np.full((len(agents), env.num_actions), 1.0 / env.num_actions)
                else:
                    rows = module_probs(modules[pid], [observations[a] for a in agents])
                for i, agent in enumerate(agents):
                    actions[agent] = int(rng.choice(env.num_actions, p=rows[i]))
            observations, _, terminations, truncations, _ = env.step(actions)
            for a in observations:
                if a not in first_seen:
                    first_seen[a] = env.current_step
                last_seen[a] = env.current_step
            active = [a for a in observations if not terminations.get(a, False) and not truncations.get(a, False)]
            if terminations.get("__all__") or truncations.get("__all__"):
                break
        # Alive at the end = got an observation this step and did not terminate. At the step cap every agent is
        # *truncated* (not terminated), so `active` (which drops truncated agents) is empty there and cannot be used.
        alive_at_end = {a for a in observations if not terminations.get(a, False)}
        for a in first_seen:
            if "predator" not in a:
                continue
            records.append(
                {
                    "sex": "predator_male" if "predator_male" in a else "predator_female",
                    "life": last_seen[a] - first_seen[a] + 1,
                    "prey_e": env.energy_from_prey[a],
                    "fruit_e": env.energy_from_fruit[a],
                    "n_prey": env.n_prey_caught[a],
                    "n_fruit": env.n_fruit_eaten[a],
                    "gift_in": env.gift_received[a],
                    "gift_out": env.gift_given[a],
                    "care_in": env.care_received[a],
                    "care_out": env.care_given[a],
                    "censored": a in alive_at_end,
                }
            )
    return records


def summarize(label, records, min_life):
    lines = []
    for sex in SEXES:
        for name, rows in (
            ("all lives", [r for r in records if r["sex"] == sex]),
            (f"lives >= {min_life} steps", [r for r in records if r["sex"] == sex and r["life"] >= min_life]),
            ("completed lives", [r for r in records if r["sex"] == sex and not r["censored"]]),
        ):
            if not rows:
                lines.append(f"{label:20} {sex[9:]:6} {name:20} n=0")
                continue
            m = lambda k: float(np.mean([r[k] for r in rows]))
            life, fruit, prey = m("life"), m("fruit_e"), m("prey_e")
            g_in, g_out, c_in, c_out = m("gift_in"), m("gift_out"), m("care_in"), m("care_out")
            own = fruit + prey
            net = own + g_in + c_in - g_out - c_out
            share = prey / own if own > 0 else float("nan")
            # individual prey shares (own foraging only), for lives with any own-forage energy
            indiv = [r["prey_e"] / (r["prey_e"] + r["fruit_e"]) for r in rows if r["prey_e"] + r["fruit_e"] > 0]
            med_share = float(np.median(indiv)) if indiv else float("nan")
            n_cens = sum(1 for r in rows if r["censored"])
            lines.append(
                f"{label:20} {sex[9:]:6} {name:20} n={len(rows):5d} life={life:6.1f} steps\n"
                f"    own foraging: fruit {fruit:6.2f} ({m('n_fruit'):5.2f} fruit) + prey {prey:6.2f} ({m('n_prey'):5.2f} prey)"
                f" = {own:6.2f}   prey share (ratio of means) {100 * share:4.1f}%, median individual {100 * med_share:4.1f}%"
                f"   [{n_cens} of {len(rows)} lives cut off at episode end]\n"
                f"    received: from mate {g_in:5.2f}, from parents {c_in:5.2f}   given away: to mate {g_out:5.2f}, to offspring {c_out:5.2f}\n"
                f"    net intake {net:6.2f} = {100.0 * net / life:5.2f} per 100 steps"
                f" (own foraging {100.0 * own / life:5.2f}/100 steps)"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", default=[], help="LABEL=path to a ray_results experiment dir")
    parser.add_argument("--checkpoint", type=int, default=29, help="checkpoint index (10 iterations per index step +1)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--min-life", type=int, default=100)
    parser.add_argument("--random", action="store_true", help="uniform-random actions (no checkpoint needed)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    runs = [(r.split("=", 1)[0], os.path.expanduser(r.split("=", 1)[1])) for r in args.run]
    output = []

    def emit(line):
        print(line, flush=True)
        output.append(line)

    for label, path in runs:
        config = json.load(open(os.path.join(path, "run_config.json")))["config_env"]
        if args.random:
            records = run_episodes(config, None, args.episodes, args.seed, random_policy=True)
            tag = f"{label} random"
        else:
            trial = sorted(glob.glob(os.path.join(path, "PPO_*/")))[0]
            modules = load_modules(os.path.join(trial, f"checkpoint_{args.checkpoint:06d}"))
            records = run_episodes(config, modules, args.episodes, args.seed)
            tag = f"{label} ckpt{args.checkpoint}"
        for line in summarize(tag, records, args.min_life):
            emit(line)
        emit("")
    if args.out:
        with open(os.path.expanduser(args.out), "w") as f:
            json.dump(output, f, indent=1)


if __name__ == "__main__":
    main()
