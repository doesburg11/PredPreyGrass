#+++++++++++++++++++++++++++++++++++++++++++++++++++
# TODO add policy for distance-based capture inspection
# - load policy modules from checkpoint
# - use policy to move predators/prey towards/away from each other
# - allow step-by-step execution to see how capture unfolds
# - visualize energy changes as well
# - expand the grid to fully utilize observation ranges
# - possibly expand grid to 25x25 to match env config
# - possibly add obstacles to test LOS effects
# - allow saving/loading of scenarios
# - consider adding a simple GUI for easier scenario editing
# --- end TODO ---
#+++++++++++++++++++++++++++++++++++++++++++++++++++
"""
Scenario Inspector for stag_hunt team-capture logic.

Usage:
    python -m predpreygrass.rllib.stag_hunt_vectorized.utils.scenario_inspector --scenario path/to/scenario.json

Scenario JSON schema (example):
{
  "grid_size": 25,
  "team_capture_margin": 0.0,
  "predators": [
    {"id": "p0", "x": 3, "y": 3, "energy": 6.0},
    {"id": "p1", "x": 4, "y": 3, "energy": 5.0}
  ],
  "prey": [
    {"id": "r0", "x": 3, "y": 4, "energy": 4.0},
    {"id": "r1", "x": 4, "y": 4, "energy": 7.0}
  ]
}

Controls:
- Space: toggle between pre-capture and post-capture view (only after a run).
- Enter: run capture simulation after edits.
- P / R: switch placement mode to predator / prey.
- Up / Down (or +/-): adjust placement energy for next agent.
- Left click: place agent on cell (respects mode and energy).
- Right click: remove any agent on cell.
- C: clear all agents.
- X: reset grid to empty.
- Esc / Q / window close: exit.

Logic mirrors the env: team capture always on, Moore neighborhood (Chebyshev ≤ 1),
sequential prey processing, energy split configurable (proportional by default); margin applied.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pygame
import textwrap

from predpreygrass.rllib.stag_hunt_vectorized.predpreygrass_rllib_env import PredPreyGrass

Coord = Tuple[int, int]

# Change this value to force a specific cell size (pixels) without needing CLI flags.
# Set to None to auto-fit the grid to the current display.
DEFAULT_CELL_SIZE: int | None = 60


@dataclass
class Agent:
    agent_id: str
    pos: Coord
    energy: float


@dataclass
class Scenario:
    grid_size: int
    predators: List[Agent]
    prey: List[Agent]
    team_capture_margin: float = 0.0

    @staticmethod
    def from_json(path: Path) -> "Scenario":
        raw = json.loads(Path(path).read_text())
        predators = [Agent(a["id"], (int(a["x"]), int(a["y"])), float(a["energy"])) for a in raw.get("predators", [])]
        prey = [Agent(a["id"], (int(a["x"]), int(a["y"])), float(a["energy"])) for a in raw.get("prey", [])]
        return Scenario(
            grid_size=int(raw.get("grid_size", 25)),
            predators=predators,
            prey=prey,
            team_capture_margin=float(raw.get("team_capture_margin", 0.0)),
        )


@dataclass
class EngagementResult:
    captured_prey: Dict[str, Dict]
    predator_energy_after: Dict[str, float]


def chebyshev(a: Coord, b: Coord) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def simulate_engagement(scn: Scenario) -> EngagementResult:
    """Run capture using the actual environment logic (no movement, just engagements)."""
    config = {
        "grid_size": scn.grid_size,
        "num_obs_channels": 4,
        "predator_obs_range": 3,
        "prey_obs_range": 3,
        "include_visibility_channel": False,
        "respect_los_for_movement": False,
        "mask_observation_with_visibility": False,
        "type_1_action_range": 3,
        "type_2_action_range": 3,
        "max_steps": 1,
        "team_capture_margin": scn.team_capture_margin,
        "initial_num_grass": 0,
    }
    env = PredPreyGrass(config)

    # Minimal state wiring for a single-step engagement run.
    env.agent_positions = {}
    env.predator_positions = {}
    env.prey_positions = {}
    env.agent_energies = {}
    env.rewards = {}
    env.terminations = {}
    env.truncations = {}
    env.observations = {}
    env.grass_positions = {}
    env.grass_energies = {}
    env.grass_agents = []
    env.current_num_grass = 0
    env._pending_infos = {}
    env.agent_event_log = {}
    env.agent_stats_live = {}
    env.agent_offspring_counts = {}
    env.agent_live_offspring_ids = {}
    env._per_agent_step_deltas = {}
    env.agents = []
    env.agent_ages = {}
    env.used_agent_ids = set()
    env.agents_just_ate = set()
    env.current_step = 0
    env.active_num_predators = len(scn.predators)
    env.active_num_prey = len(scn.prey)

    # Fresh grid for overlay updates.
    channels = env.num_obs_channels + (1 if env.include_visibility_channel else 0)
    env.grid_world_state = __import__("numpy").zeros((channels, scn.grid_size, scn.grid_size), dtype=float)

    # Place agents using scenario IDs directly.
    for pred in scn.predators:
        aid = pred.agent_id
        env.agent_positions[aid] = pred.pos
        env.predator_positions[aid] = pred.pos
        env.agent_energies[aid] = pred.energy
        env.agents.append(aid)
        env.agent_ages[aid] = 0
        env.used_agent_ids.add(aid)
        env.agent_stats_live[aid] = {
            "offspring_ids": [],
            "offspring_count": 0,
            "times_ate": 0,
            "energy_gained": 0.0,
            "cumulative_reward": 0.0,
        }
        env.agent_offspring_counts[aid] = 0
        env.agent_live_offspring_ids[aid] = []

    for prey in scn.prey:
        aid = prey.agent_id
        env.agent_positions[aid] = prey.pos
        env.prey_positions[aid] = prey.pos
        env.agent_energies[aid] = prey.energy
        env.agents.append(aid)
        env.agent_ages[aid] = 0
        env.used_agent_ids.add(aid)
        env.agent_stats_live[aid] = {
            "offspring_ids": [],
            "offspring_count": 0,
            "times_ate": 0,
            "energy_gained": 0.0,
            "cumulative_reward": 0.0,
        }
        env.agent_offspring_counts[aid] = 0
        env.agent_live_offspring_ids[aid] = []

    captured: Dict[str, Dict] = {}

    # Wrap team-capture to record helpers/share while using env logic.
    original_team_capture = env._handle_team_capture

    def wrapped_team_capture(prey_id):
        prey_pos = tuple(env.agent_positions[prey_id])
        helpers = env._predators_in_moore_neighborhood(prey_pos)
        prey_energy = env.agent_energies[prey_id]
        ok = original_team_capture(prey_id)
        if ok:
            share = prey_energy / len(helpers) if helpers else 0.0
            captured[prey_id] = {
                "pos": prey_pos,
                "helpers": helpers,
                "energy_share": share,
                "prey_energy": prey_energy,
            }
        return ok

    env._handle_team_capture = wrapped_team_capture  # type: ignore[attr-defined]

    # Process prey engagements in env order.
    for prey_id in list(env.prey_positions.keys()):
        env._handle_prey_engagement(prey_id)

    predator_energy_after = {pid: env.agent_energies[pid] for pid in env.predator_positions}
    return EngagementResult(captured_prey=captured, predator_energy_after=predator_energy_after)


class ScenarioViewer:
    def __init__(self, scn: Scenario, result: EngagementResult, cell_override: int | None = DEFAULT_CELL_SIZE):
        self.scn = scn
        self.result = result
        pygame.init()
        info = pygame.display.Info()
        status_panel = 520
        padding = 60
        avail_w = max(200, info.current_w - padding - status_panel)
        avail_h = max(200, info.current_h - padding)
        fit_cell = min(avail_w // scn.grid_size, avail_h // scn.grid_size)
        if fit_cell <= 0:
            fit_cell = 24
        if cell_override is not None:
            desired = max(24, cell_override)
            self.cell = desired
        else:
            self.cell = max(24, fit_cell)
        self.status_width = status_panel
        w = scn.grid_size * self.cell + self.status_width
        h = scn.grid_size * self.cell
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Scenario Inspector — space toggle view; P/R modes; +/- energy; Enter recompute")
        self.font = pygame.font.SysFont(None, max(18, self.cell // 2))
        self.status_font = pygame.font.SysFont(None, max(36, self.font.get_height() * 2))
        self.show_post = False
        self.place_mode = "predator"
        self.place_energy = 5.0
        self.pred_counter = 0
        self.prey_counter = 0
        self.selected: Coord | None = None
        self.needs_run = False
        self.status_msg = "Edit the grid, then press Enter to run"

    def draw_grid(self):
        g = self.scn.grid_size
        for x in range(g):
            for y in range(g):
                rect = pygame.Rect(x * self.cell, y * self.cell, self.cell, self.cell)
                pygame.draw.rect(self.screen, (230, 230, 230), rect, 1)

    def draw_agents(self):
        # prey first (blue), then predators (red)
        captured = self.result.captured_prey if self.show_post else {}
        for prey in self.scn.prey:
            color = (120, 120, 120) if prey.agent_id in captured else (50, 100, 255)
            self._draw_agent(prey.pos, color, prey.agent_id, prey.energy)
            if prey.agent_id in captured and self.show_post:
                helpers = ",".join(captured[prey.agent_id]["helpers"])
                # Render helper list just above the cell bottom for readability.
                self._draw_text(helpers, prey.pos, offset=(0, self.cell - 28), color=(30, 30, 30))
        for pred in self.scn.predators:
            energy = self.result.predator_energy_after.get(pred.agent_id, pred.energy) if self.show_post else pred.energy
            self._draw_agent(pred.pos, (200, 40, 40), pred.agent_id, energy)
        if self.selected:
            x, y = self.selected
            rect = pygame.Rect(x * self.cell, y * self.cell, self.cell, self.cell)
            pygame.draw.rect(self.screen, (0, 180, 0), rect, 3)

    def _draw_agent(self, pos: Coord, color, label: str, energy: float):
        x, y = pos
        cx = x * self.cell + self.cell // 2
        cy = y * self.cell + self.cell // 2
        pygame.draw.circle(self.screen, color, (cx, cy), self.cell // 3)
        self._draw_text(f"{label}\n{energy:.1f}", pos, color=(0, 0, 0))

    def _draw_text(self, text: str, pos: Coord, offset=(0, 0), color=(0, 0, 0)):
        lines = text.split("\n")
        x, y = pos
        base_x = x * self.cell + 5 + offset[0]
        base_y = y * self.cell + 5 + offset[1]
        line_height = self.font.get_linesize()
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, color)
            self.screen.blit(surf, (base_x, base_y + i * line_height))

    def loop(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.needs_run:
                            self.show_post = not self.show_post
                        else:
                            self.show_post = False
                            self.status_msg = "Pending changes — press Enter to run"
                    elif event.key == pygame.K_RETURN:
                        self.result = simulate_engagement(self.scn)
                        self.show_post = True
                        self.needs_run = False
                        self.status_msg = "Ran simulation — Space toggles view"
                    elif event.key == pygame.K_p:
                        self.place_mode = "predator"
                    elif event.key == pygame.K_r:
                        self.place_mode = "prey"
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
                        self._change_energy(1.0)
                    elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
                        self._change_energy(-1.0)
                    elif event.key == pygame.K_c:
                        self._reset_grid(status="Cleared — set up and press Enter to run")
                    elif event.key == pygame.K_x:
                        self._reset_grid(status="Reset to empty grid — set up and press Enter to run")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    grid_pixels = self.scn.grid_size * self.cell
                    if event.pos[0] >= grid_pixels or event.pos[1] >= grid_pixels:
                        continue
                    x, y = event.pos[0] // self.cell, event.pos[1] // self.cell
                    if 0 <= x < self.scn.grid_size and 0 <= y < self.scn.grid_size:
                        occupant = self._agent_at((x, y))
                        if event.button == 1:
                            if occupant:
                                self.selected = (x, y)
                                self.place_mode = "predator" if occupant in self.scn.predators else "prey"
                                self.place_energy = occupant.energy
                                self.status_msg = f"Selected {occupant.agent_id}"
                            else:
                                self._place_agent((x, y))
                                self.selected = (x, y)
                                self.needs_run = True
                                self.show_post = False
                                self.status_msg = "Pending changes — press Enter to run"
                        elif event.button == 3:
                            self._remove_agent((x, y))
                            if self.selected == (x, y):
                                self.selected = None
                            self.needs_run = True
                            self.show_post = False
                            self.status_msg = "Pending changes — press Enter to run"
            self.screen.fill((255, 255, 255))
            self.draw_grid()
            self.draw_agents()
            self._draw_status()
            pygame.display.flip()
            clock.tick(30)
        pygame.quit()

    def _place_agent(self, pos: Coord):
        # Clear existing occupant then place new agent of selected type.
        self._remove_agent(pos)
        if self.place_mode == "predator":
            agent_id = f"p{self.pred_counter}"
            self.pred_counter += 1
            self.scn.predators.append(Agent(agent_id, pos, self.place_energy))
        else:
            agent_id = f"r{self.prey_counter}"
            self.prey_counter += 1
            self.scn.prey.append(Agent(agent_id, pos, self.place_energy))

    def _remove_agent(self, pos: Coord):
        self.scn.predators = [a for a in self.scn.predators if a.pos != pos]
        self.scn.prey = [a for a in self.scn.prey if a.pos != pos]

    def _agent_at(self, pos: Coord):
        for a in self.scn.predators:
            if a.pos == pos:
                return a
        for a in self.scn.prey:
            if a.pos == pos:
                return a
        return None

    def _draw_status(self):
        grid_px = self.scn.grid_size * self.cell
        panel_rect = pygame.Rect(grid_px, 0, self.status_width, grid_px)
        pygame.draw.rect(self.screen, (245, 245, 245), panel_rect)
        pygame.draw.rect(self.screen, (210, 210, 210), panel_rect, 2)
        info_lines = [
            f"Mode: {self.place_mode}",
            f"Energy: {self.place_energy:.1f}",
            "Enter: run",
            "Space: toggle",
            "P/R: mode",
            "+/- or Up/Down: energy",
            "Left click: place",
            "Right click: remove",
            "C: clear",
            "X: reset",
        ]
        x = grid_px + 10
        y = 10
        font = self.status_font
        line_height = font.get_linesize()
        for line in info_lines:
            surf = font.render(line, True, (0, 0, 0))
            self.screen.blit(surf, (x, y))
            y += line_height + 2
        y += line_height
        for line in textwrap.wrap(self.status_msg, width=20):
            surf = font.render(line, True, (0, 0, 0))
            self.screen.blit(surf, (x, y))
            y += line_height

    def _reset_grid(self, status: str):
        self.scn.predators.clear()
        self.scn.prey.clear()
        self.pred_counter = 0
        self.prey_counter = 0
        self.result = simulate_engagement(self.scn)
        self.show_post = False
        self.needs_run = True
        self.status_msg = status

    def _change_energy(self, delta: float):
        self.place_energy = round(max(0.1, self.place_energy + delta), 2)
        # Only update the selected agent (if it matches current mode).
        updated = False
        if self.selected:
            if self.place_mode == "predator":
                for agent in self.scn.predators:
                    if agent.pos == self.selected:
                        agent.energy = self.place_energy
                        updated = True
                        break
            else:
                for agent in self.scn.prey:
                    if agent.pos == self.selected:
                        agent.energy = self.place_energy
                        updated = True
                        break
        if updated:
            self.needs_run = True
            self.show_post = False
            self.status_msg = "Energy adjusted — press Enter to run"
        else:
            self.status_msg = "No agent selected; click an agent to adjust"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize team-capture engagement on a small scenario.")
    ap.add_argument("--scenario", required=False, help="Path to scenario JSON. If omitted, a built-in demo is used.")
    ap.add_argument(
        "--cell-size",
        type=int,
        default=DEFAULT_CELL_SIZE,
        help="Override grid cell size (pixels); defaults to DEFAULT_CELL_SIZE in this file.",
    )
    return ap.parse_args()


def demo_scenario() -> Scenario:
    # Start with an empty grid for manual setup.
    demo = {
        "grid_size": 25,
        "predators": [],
        "prey": [],
        "team_capture_margin": 0.0,
    }
    tmp = Path("/tmp/demo_scenario.json")
    tmp.write_text(json.dumps(demo, indent=2))
    return Scenario.from_json(tmp)


def main():
    args = parse_args()
    scn = Scenario.from_json(Path(args.scenario)) if args.scenario else demo_scenario()
    result = simulate_engagement(scn)
    print("Captured prey:")
    if not result.captured_prey:
        print("  None")
    else:
        for pid, rec in result.captured_prey.items():
            helpers = ",".join(rec["helpers"])
            print(f"  {pid} at {rec['pos']} eaten by [{helpers}] share={rec['energy_share']:.2f}")
    viewer = ScenarioViewer(scn, result, cell_override=args.cell_size)
    viewer.loop()


if __name__ == "__main__":
    main()
