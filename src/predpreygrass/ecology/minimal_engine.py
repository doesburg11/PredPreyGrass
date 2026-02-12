"""
Minimal ecological engine with heritable traits and no learning.
Pure-Python, grid-based, discrete ticks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import random


Vec2 = Tuple[int, int]

# Edit these for quick runs without CLI arguments.
RUN_SETTINGS = {
    "steps": 1000,
    "log_every": 10,
    "seed": 1,
    "plot_path": None,  # e.g., "/tmp/ecology.png"
    "render": True,
    "render_cell_size": 16,
    "render_fps": 20,
    "live_plot": True,
    "live_plot_interval": 5,
}

# Optional config overrides (leave empty to use WorldConfig defaults).
CONFIG_OVERRIDES: Dict[str, object] = {
    # "width": 30,
    # "height": 30,
    # "initial_prey": 20,
    # "initial_predators": 6,
    # "mutation_sigma": 0.1,
}

# Optional config file (JSON). If present, it seeds WorldConfig before overrides.
CONFIG_PATH = Path(__file__).with_name("minimal_config.json")


@dataclass
class Agent:
    agent_id: int
    kind: str  # "prey" or "predator"
    x: int
    y: int
    energy: float
    traits: Dict[str, float]
    age: int = 0
    cooldown: int = 0


@dataclass
class WorldState:
    width: int
    height: int
    grass: List[List[float]]
    agents: List[Agent]
    tick: int = 0
    next_id: int = 0


@dataclass
class WorldConfig:
    width: int = 30
    height: int = 30
    gmax: float = 5.0
    grass_regrow_prob: float = 0.05
    grass_regrow_amount: float = 1.0
    initial_grass_prob: float = 0.2
    initial_grass_amount: float = 3.0
    initial_prey: int = 20
    initial_predators: int = 6
    prey_energy_init: float = 12.0
    predator_energy_init: float = 20.0
    prey_eat_amount: float = 2.0
    prey_energy_gain: float = 3.0
    predator_kill_gain: float = 30.0
    predator_fail_cost: float = 2.0
    base_metabolism_prey: float = 0.2
    base_metabolism_predator: float = 0.3
    speed_cost: float = 0.08
    vision_cost: float = 0.04
    move_cost: float = 0.05
    repro_threshold_prey: float = 25.0
    repro_threshold_predator: float = 40.0
    repro_cooldown: int = 5
    parent_energy_share: float = 0.6
    mutation_sigma: float = 0.1
    max_age: Optional[int] = 200
    num_obs_channels: int = 4
    prey_obs_range: int = 5
    predator_obs_range: int = 7
    trait_ranges_prey: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "speed": (0.5, 3.0),
            "vision": (1.0, 8.0),
            "metabolism": (0.05, 0.4),
        }
    )
    trait_ranges_predator: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "speed": (0.6, 3.2),
            "vision": (1.0, 9.0),
            "metabolism": (0.08, 0.6),
            "attack_power": (0.5, 2.0),
        }
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rng_steps(speed: float, rng: random.Random) -> int:
    steps = int(speed)
    frac = speed - steps
    if rng.random() < frac:
        steps += 1
    return steps


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _chebyshev_dist(a: Vec2, b: Vec2) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _random_step(rng: random.Random) -> Vec2:
    return rng.choice(
        [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    )


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _find_nearest_agent(
    agent: Agent, agents: List[Agent], target_kind: str, vision: int
) -> Optional[Agent]:
    best = None
    best_dist = None
    ax, ay = agent.x, agent.y
    for other in agents:
        if other.kind != target_kind:
            continue
        dist = _chebyshev_dist((ax, ay), (other.x, other.y))
        if dist == 0 or dist > vision:
            continue
        if best_dist is None or dist < best_dist:
            best = other
            best_dist = dist
    return best


def _find_best_grass(
    agent: Agent, grass: List[List[float]], vision: int
) -> Optional[Tuple[int, int, float]]:
    best = None
    ax, ay = agent.x, agent.y
    height = len(grass)
    width = len(grass[0]) if height else 0
    for dx in range(-vision, vision + 1):
        for dy in range(-vision, vision + 1):
            nx = ax + dx
            ny = ay + dy
            if not _in_bounds(nx, ny, width, height):
                continue
            amount = grass[ny][nx]
            if amount <= 0:
                continue
            if best is None or amount > best[2]:
                best = (nx, ny, amount)
    return best


def _obs_clip(
    x: int, y: int, observation_range: int, width: int, height: int
) -> Tuple[int, int, int, int, int, int, int, int]:
    offset = (observation_range - 1) // 2
    xld, xhd = x - offset, x + offset
    yld, yhd = y - offset, y + offset
    xlo = max(xld, 0)
    xhi = min(xhd, width - 1)
    ylo = max(yld, 0)
    yhi = min(yhd, height - 1)
    xolo = abs(min(xld, 0))
    yolo = abs(min(yld, 0))
    xohi = xolo + (xhi - xlo)
    yohi = yolo + (yhi - ylo)
    return xlo, xhi + 1, ylo, yhi + 1, xolo, xohi + 1, yolo, yohi + 1


def _random_traits(ranges: Dict[str, Tuple[float, float]], rng: random.Random) -> Dict[str, float]:
    return {name: rng.uniform(lo, hi) for name, (lo, hi) in ranges.items()}


def _mutate_traits(
    traits: Dict[str, float],
    ranges: Dict[str, Tuple[float, float]],
    sigma: float,
    rng: random.Random,
) -> Dict[str, float]:
    mutated = {}
    for name, value in traits.items():
        low, high = ranges[name]
        new_value = value + rng.gauss(0.0, sigma)
        mutated[name] = clamp(new_value, low, high)
    return mutated


def spawn_agent(
    state: WorldState, kind: str, traits: Dict[str, float], pos: Vec2, energy: float
) -> Agent:
    agent = Agent(
        agent_id=state.next_id,
        kind=kind,
        x=pos[0],
        y=pos[1],
        energy=energy,
        traits=traits,
    )
    state.next_id += 1
    state.agents.append(agent)
    return agent


def sense(agent: Agent, state: WorldState, config: WorldConfig) -> Dict[str, Optional[Tuple[int, int, float]]]:
    vision = int(round(agent.traits.get("vision", 1.0)))
    nearest_prey = _find_nearest_agent(agent, state.agents, "prey", vision)
    nearest_predator = _find_nearest_agent(agent, state.agents, "predator", vision)
    best_grass = _find_best_grass(agent, state.grass, vision)
    prey_dist = None if nearest_prey is None else _chebyshev_dist((agent.x, agent.y), (nearest_prey.x, nearest_prey.y))
    pred_dist = None if nearest_predator is None else _chebyshev_dist((agent.x, agent.y), (nearest_predator.x, nearest_predator.y))
    return {
        "nearest_prey": None
        if nearest_prey is None
        else (nearest_prey.x - agent.x, nearest_prey.y - agent.y, float(prey_dist)),
        "nearest_predator": None
        if nearest_predator is None
        else (nearest_predator.x - agent.x, nearest_predator.y - agent.y, float(pred_dist)),
        "best_grass": best_grass,
        "local_grass": (agent.x, agent.y, state.grass[agent.y][agent.x]),
    }


def sense_grid(agent: Agent, state: WorldState, config: WorldConfig) -> List[List[List[float]]]:
    """
    Grid observation compatible with the RLlib layout:
    channel 0: boundary mask (1 outside world, 0 inside)
    channel 1: predator energy (or presence)
    channel 2: prey energy (or presence)
    channel 3: grass amount
    """
    if config.num_obs_channels < 4:
        raise ValueError("num_obs_channels must be >= 4 for RLlib-compatible layout")
    observation_range = (
        config.predator_obs_range if agent.kind == "predator" else config.prey_obs_range
    )
    xlo, xhi, ylo, yhi, xolo, xohi, yolo, yohi = _obs_clip(
        agent.x, agent.y, observation_range, state.width, state.height
    )
    obs = [
        [[0.0 for _ in range(observation_range)] for _ in range(observation_range)]
        for _ in range(config.num_obs_channels)
    ]
    # Boundary mask defaults to 1 outside world; set inside to 0
    for i in range(observation_range):
        for j in range(observation_range):
            obs[0][i][j] = 1.0
    for i in range(xolo, xohi):
        for j in range(yolo, yohi):
            obs[0][i][j] = 0.0

    # Fill local window with entities + grass
    for gx in range(xlo, xhi):
        for gy in range(ylo, yhi):
            ox = xolo + (gx - xlo)
            oy = yolo + (gy - ylo)
            obs[3][ox][oy] = state.grass[gy][gx]

    for other in state.agents:
        if other.x < xlo or other.x >= xhi or other.y < ylo or other.y >= yhi:
            continue
        ox = xolo + (other.x - xlo)
        oy = yolo + (other.y - ylo)
        channel = 1 if other.kind == "predator" else 2
        obs[channel][ox][oy] = max(obs[channel][ox][oy], other.energy)

    return obs


def _move_agent(agent: Agent, dx: int, dy: int, width: int, height: int) -> None:
    nx = clamp(agent.x + dx, 0, width - 1)
    ny = clamp(agent.y + dy, 0, height - 1)
    agent.x = int(nx)
    agent.y = int(ny)


def _kill_probability(predator: Agent, prey: Agent) -> float:
    attack = predator.traits.get("attack_power", 1.0)
    pred_speed = predator.traits.get("speed", 1.0)
    prey_speed = prey.traits.get("speed", 1.0)
    score = attack + 0.5 * pred_speed - 0.5 * prey_speed
    prob = 1.0 / (1.0 + math.exp(-score))
    return clamp(prob, 0.05, 0.95)


def _choose_spawn_pos(
    parent: Agent, occupied: set[Tuple[int, int]], width: int, height: int, rng: random.Random
) -> Optional[Vec2]:
    candidates = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            nx = parent.x + dx
            ny = parent.y + dy
            if not _in_bounds(nx, ny, width, height):
                continue
            if (nx, ny) not in occupied:
                candidates.append((nx, ny))
    if not candidates:
        return None
    return rng.choice(candidates)


def step(state: WorldState, config: WorldConfig, rng: random.Random) -> Dict[str, int]:
    births = 0
    deaths = 0
    state.tick += 1

    # Grass regrowth
    for y in range(state.height):
        for x in range(state.width):
            if state.grass[y][x] >= config.gmax:
                continue
            if rng.random() < config.grass_regrow_prob:
                state.grass[y][x] = clamp(
                    state.grass[y][x] + config.grass_regrow_amount, 0.0, config.gmax
                )

    # Movement planning and execution
    steps_moved: Dict[int, int] = {}
    for agent in state.agents:
        steps = _rng_steps(agent.traits.get("speed", 1.0), rng)
        steps_moved[agent.agent_id] = steps
        if steps <= 0:
            continue

        vision = int(round(agent.traits.get("vision", 1.0)))
        dx, dy = 0, 0

        if agent.kind == "prey":
            nearest_predator = _find_nearest_agent(agent, state.agents, "predator", vision)
            if nearest_predator is not None:
                dx = _sign(agent.x - nearest_predator.x)
                dy = _sign(agent.y - nearest_predator.y)
            else:
                best_grass = _find_best_grass(agent, state.grass, vision)
                if best_grass is not None:
                    dx = _sign(best_grass[0] - agent.x)
                    dy = _sign(best_grass[1] - agent.y)
                else:
                    dx, dy = _random_step(rng)
        else:
            nearest_prey = _find_nearest_agent(agent, state.agents, "prey", vision)
            if nearest_prey is not None:
                dx = _sign(nearest_prey.x - agent.x)
                dy = _sign(nearest_prey.y - agent.y)
            else:
                dx, dy = _random_step(rng)

        for _ in range(steps):
            _move_agent(agent, dx, dy, state.width, state.height)

    # Energy costs
    dead_ids = set()
    for agent in state.agents:
        speed = agent.traits.get("speed", 1.0)
        vision = agent.traits.get("vision", 1.0)
        metabolism = agent.traits.get("metabolism", 0.1)
        move_steps = steps_moved.get(agent.agent_id, 0)
        if agent.kind == "prey":
            base = config.base_metabolism_prey
        else:
            base = config.base_metabolism_predator
        cost = (
            base
            + metabolism
            + config.speed_cost * speed * speed
            + config.vision_cost * vision
            + config.move_cost * move_steps
        )
        agent.energy -= cost
        if agent.energy <= 0:
            dead_ids.add(agent.agent_id)

    # Prey eat grass
    for agent in state.agents:
        if agent.kind != "prey":
            continue
        if agent.agent_id in dead_ids:
            continue
        x, y = agent.x, agent.y
        if state.grass[y][x] <= 0:
            continue
        amount = min(config.prey_eat_amount, state.grass[y][x])
        state.grass[y][x] -= amount
        agent.energy += config.prey_energy_gain * amount

    # Predator kills
    cell_prey: Dict[Vec2, List[Agent]] = {}
    cell_pred: Dict[Vec2, List[Agent]] = {}
    for agent in state.agents:
        if agent.agent_id in dead_ids:
            continue
        key = (agent.x, agent.y)
        if agent.kind == "prey":
            cell_prey.setdefault(key, []).append(agent)
        else:
            cell_pred.setdefault(key, []).append(agent)

    for cell, predators in cell_pred.items():
        prey_list = cell_prey.get(cell, [])
        if not prey_list:
            continue
        for predator in predators:
            if not prey_list:
                break
            prey = rng.choice(prey_list)
            if prey.agent_id in dead_ids:
                continue
            if rng.random() < _kill_probability(predator, prey):
                dead_ids.add(prey.agent_id)
                predator.energy += config.predator_kill_gain
                prey_list.remove(prey)
            else:
                predator.energy -= config.predator_fail_cost

    # Cull dead before reproduction
    survivors = []
    for agent in state.agents:
        if agent.agent_id in dead_ids:
            deaths += 1
            continue
        if agent.energy <= 0:
            deaths += 1
            continue
        if config.max_age is not None and agent.age >= config.max_age:
            deaths += 1
            continue
        survivors.append(agent)
    state.agents = survivors

    # Reproduction
    occupied = {(agent.x, agent.y) for agent in state.agents}
    new_agents: List[Agent] = []
    for agent in state.agents:
        if agent.cooldown > 0:
            agent.cooldown -= 1
            continue
        threshold = (
            config.repro_threshold_prey
            if agent.kind == "prey"
            else config.repro_threshold_predator
        )
        if agent.energy < threshold:
            continue
        spawn_pos = _choose_spawn_pos(agent, occupied, state.width, state.height, rng)
        if spawn_pos is None:
            continue
        if agent.kind == "prey":
            ranges = config.trait_ranges_prey
        else:
            ranges = config.trait_ranges_predator
        child_traits = _mutate_traits(agent.traits, ranges, config.mutation_sigma, rng)
        child_energy = agent.energy * (1.0 - config.parent_energy_share)
        agent.energy *= config.parent_energy_share
        agent.cooldown = config.repro_cooldown
        child = Agent(
            agent_id=state.next_id,
            kind=agent.kind,
            x=spawn_pos[0],
            y=spawn_pos[1],
            energy=child_energy,
            traits=child_traits,
        )
        state.next_id += 1
        new_agents.append(child)
        occupied.add(spawn_pos)
        births += 1

    state.agents.extend(new_agents)

    # Age update
    for agent in state.agents:
        agent.age += 1

    return {"births": births, "deaths": deaths}


def compute_stats(state: WorldState, config: WorldConfig) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for kind, ranges in [
        ("prey", config.trait_ranges_prey),
        ("predator", config.trait_ranges_predator),
    ]:
        values = {name: [] for name in ranges.keys()}
        for agent in state.agents:
            if agent.kind != kind:
                continue
            for name in values.keys():
                values[name].append(agent.traits.get(name, 0.0))
        stats[kind] = {}
        for name, vals in values.items():
            if not vals:
                stats[kind][f"{name}_mean"] = 0.0
                stats[kind][f"{name}_var"] = 0.0
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            stats[kind][f"{name}_mean"] = mean
            stats[kind][f"{name}_var"] = var
        stats[kind]["count"] = sum(1 for agent in state.agents if agent.kind == kind)
    stats["grass"] = {
        "mean": sum(sum(row) for row in state.grass) / (state.width * state.height)
    }
    return stats


def load_config(path: str | Path) -> WorldConfig:
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = WorldConfig()
    for field_info in fields(WorldConfig):
        name = field_info.name
        if name not in data:
            continue
        value = data[name]
        if name in ("trait_ranges_prey", "trait_ranges_predator"):
            cleaned = {}
            for key, bounds in value.items():
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"Invalid range for {name}.{key}: {bounds}")
                cleaned[key] = (float(bounds[0]), float(bounds[1]))
            value = cleaned
        setattr(cfg, name, value)
    return cfg


def plot_history(history: Dict[str, List[float]], out_path: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    ticks = history.get("tick", [])
    prey = history.get("prey_count", [])
    pred = history.get("predator_count", [])
    grass = history.get("grass_mean", [])
    prey_trait_keys = sorted([k for k in history.keys() if k.startswith("prey_") and k.endswith("_mean")])
    pred_trait_keys = sorted([k for k in history.keys() if k.startswith("predator_") and k.endswith("_mean")])

    rows = 2 + (1 if prey_trait_keys else 0) + (1 if pred_trait_keys else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(ticks, prey, label="prey")
    ax.plot(ticks, pred, label="predator")
    ax.set_ylabel("population")
    ax.legend()

    ax = axes[1]
    ax.plot(ticks, grass, label="grass mean")
    ax.set_ylabel("grass")
    ax.legend()

    idx = 2
    if prey_trait_keys:
        ax = axes[idx]
        for key in prey_trait_keys:
            label = key.replace("prey_", "").replace("_mean", "")
            ax.plot(ticks, history.get(key, []), label=label)
        ax.set_ylabel("prey traits (mean)")
        ax.legend()
        idx += 1

    if pred_trait_keys:
        ax = axes[idx]
        for key in pred_trait_keys:
            label = key.replace("predator_", "").replace("_mean", "")
            ax.plot(ticks, history.get(key, []), label=label)
        ax.set_ylabel("predator traits (mean)")
        ax.legend()

    axes[-1].set_xlabel("tick")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
    else:
        plt.show()


class LivePlotter:
    def __init__(self, history: Dict[str, List[float]], config: WorldConfig):
        import os
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required for live plotting") from exc

        self.plt = plt
        if not os.environ.get("DISPLAY"):
            print("LivePlotter disabled: no DISPLAY found (headless session).")
            self.alive = False
            return
        self.plt.ion()
        self.prey_trait_keys = [f"prey_{name}_mean" for name in config.trait_ranges_prey.keys()]
        self.pred_trait_keys = [f"predator_{name}_mean" for name in config.trait_ranges_predator.keys()]

        rows = 2 + (1 if self.prey_trait_keys else 0) + (1 if self.pred_trait_keys else 0)
        self.fig, self.axes = self.plt.subplots(rows, 1, figsize=(10, 3 * rows), sharex=True)
        if rows == 1:
            self.axes = [self.axes]

        self.lines = {}
        ax = self.axes[0]
        self.lines["prey_count"], = ax.plot([], [], label="prey")
        self.lines["predator_count"], = ax.plot([], [], label="predator")
        ax.set_ylabel("population")
        ax.legend()

        ax = self.axes[1]
        self.lines["grass_mean"], = ax.plot([], [], label="grass mean")
        ax.set_ylabel("grass")
        ax.legend()

        idx = 2
        if self.prey_trait_keys:
            ax = self.axes[idx]
            for key in self.prey_trait_keys:
                label = key.replace("prey_", "").replace("_mean", "")
                self.lines[key], = ax.plot([], [], label=label)
            ax.set_ylabel("prey traits (mean)")
            ax.legend()
            idx += 1

        if self.pred_trait_keys:
            ax = self.axes[idx]
            for key in self.pred_trait_keys:
                label = key.replace("predator_", "").replace("_mean", "")
                self.lines[key], = ax.plot([], [], label=label)
            ax.set_ylabel("predator traits (mean)")
            ax.legend()

        self.axes[-1].set_xlabel("tick")
        self.fig.tight_layout()
        self.plt.show(block=False)
        self.alive = True

    def update(self, history: Dict[str, List[float]]) -> bool:
        if not self.alive:
            return False
        try:
            ticks = history.get("tick", [])
            if not ticks:
                return True
            for key, line in self.lines.items():
                series = history.get(key, [])
                line.set_data(ticks, series)

            xmax = max(ticks[-1], 1)
            for ax in self.axes:
                ax.set_xlim(0, xmax)

            # Autoscale y using current data
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
            return True
        except Exception:
            self.alive = False
            return False


def init_world(config: WorldConfig, rng: random.Random) -> WorldState:
    grass = [[0.0 for _ in range(config.width)] for _ in range(config.height)]
    for y in range(config.height):
        for x in range(config.width):
            if rng.random() < config.initial_grass_prob:
                grass[y][x] = clamp(config.initial_grass_amount, 0.0, config.gmax)

    state = WorldState(width=config.width, height=config.height, grass=grass, agents=[])
    for _ in range(config.initial_prey):
        pos = (rng.randrange(config.width), rng.randrange(config.height))
        traits = _random_traits(config.trait_ranges_prey, rng)
        spawn_agent(state, "prey", traits, pos, config.prey_energy_init)
    for _ in range(config.initial_predators):
        pos = (rng.randrange(config.width), rng.randrange(config.height))
        traits = _random_traits(config.trait_ranges_predator, rng)
        spawn_agent(state, "predator", traits, pos, config.predator_energy_init)
    return state


def run_simulation(
    steps: int = 200,
    log_every: int = 10,
    seed: Optional[int] = 1,
    config: Optional[WorldConfig] = None,
    collect_history: bool = False,
    render: bool = False,
    render_cell_size: int = 16,
    render_fps: int = 20,
    live_plot: bool = False,
    live_plot_interval: int = 5,
) -> Dict[str, List[float]]:
    if live_plot:
        collect_history = True
    cfg = config or WorldConfig()
    rng = random.Random(seed)
    state = init_world(cfg, rng)
    history: Dict[str, List[float]] = {
        "tick": [],
        "prey_count": [],
        "predator_count": [],
        "grass_mean": [],
        "births": [],
        "deaths": [],
    }
    renderer = None
    plotter = None
    if render:
        try:
            from predpreygrass.ecology.pygame_renderer import PyGameRenderer
        except Exception as exc:
            raise RuntimeError("pygame is required for rendering") from exc
        renderer = PyGameRenderer(state.width, state.height, cell_size=render_cell_size, fps=render_fps)
    if live_plot:
        plotter = LivePlotter(history, cfg)

    for step_idx in range(steps):
        events = step(state, cfg, rng)
        stats = compute_stats(state, cfg)
        if collect_history:
            history["tick"].append(state.tick)
            history["prey_count"].append(stats["prey"]["count"])
            history["predator_count"].append(stats["predator"]["count"])
            history["grass_mean"].append(stats["grass"]["mean"])
            history["births"].append(events["births"])
            history["deaths"].append(events["deaths"])
            for trait_name in cfg.trait_ranges_prey.keys():
                key = f"prey_{trait_name}_mean"
                history.setdefault(key, []).append(stats["prey"][f"{trait_name}_mean"])
            for trait_name in cfg.trait_ranges_predator.keys():
                key = f"predator_{trait_name}_mean"
                history.setdefault(key, []).append(stats["predator"][f"{trait_name}_mean"])
        if log_every > 0 and (step_idx % log_every == 0 or step_idx == steps - 1):
            prey = stats["prey"]["count"]
            pred = stats["predator"]["count"]
            grass = stats["grass"]["mean"]
            print(
                f"t={state.tick:04d} prey={prey:3d} pred={pred:3d} "
                f"grass={grass:4.2f} births={events['births']:2d} deaths={events['deaths']:2d}"
            )
        if renderer:
            if not renderer.update(state, cfg, state.tick, stats=stats):
                break
        if plotter and live_plot_interval > 0 and (step_idx % live_plot_interval == 0):
            if not plotter.update(history):
                plotter = None
    if renderer:
        renderer.close()
    return history if collect_history else {}


def build_config(overrides: Optional[Dict[str, object]] = None) -> WorldConfig:
    if CONFIG_PATH.exists():
        cfg = load_config(CONFIG_PATH)
    else:
        cfg = WorldConfig()
    if overrides:
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown WorldConfig field: {key}")
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    cfg = build_config(CONFIG_OVERRIDES)
    history = run_simulation(
        steps=int(RUN_SETTINGS["steps"]),
        log_every=int(RUN_SETTINGS["log_every"]),
        seed=int(RUN_SETTINGS["seed"]),
        config=cfg,
        collect_history=bool(RUN_SETTINGS.get("plot_path")),
        render=bool(RUN_SETTINGS.get("render")),
        render_cell_size=int(RUN_SETTINGS.get("render_cell_size", 16)),
        render_fps=int(RUN_SETTINGS.get("render_fps", 20)),
        live_plot=bool(RUN_SETTINGS.get("live_plot")),
        live_plot_interval=int(RUN_SETTINGS.get("live_plot_interval", 5)),
    )
    if RUN_SETTINGS.get("plot_path"):
        plot_history(history, RUN_SETTINGS["plot_path"])


if __name__ == "__main__":
    main()
