"""Live random-policy viewer for article-task reconstruction environments.

This viewer is intentionally separate from RLlib training. It is meant for
quick visual inspection of the reconstructed Clamity and Allelopathy dynamics.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import pygame

from predpreygrass.non_evolutionary.malthusian_rl.article_tasks import (
    ArticleAllelopathyEnv,
    ArticleClamityEnv,
)
from predpreygrass.non_evolutionary.malthusian_rl.config.config_article_protocol import (
    make_article_task_config,
)


SPECIES_COLORS = [
    (61, 110, 175),
    (214, 95, 67),
    (66, 151, 94),
    (145, 91, 171),
    (221, 171, 67),
    (77, 157, 173),
]
SHRUB_COLORS = {
    -1: (242, 239, 229),
    0: (82, 151, 88),
    1: (169, 104, 176),
}
BACKGROUND = (248, 247, 242)
GRID_LINE = (218, 214, 202)
TEXT = (34, 39, 43)
SHELL = (112, 97, 81)
NUTRIENT = (230, 186, 63)
SETTLED_CENTER = (35, 35, 35)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["allelopathy", "clamity"], default="allelopathy")
    parser.add_argument("--variant", choices=["biased", "unbiased"], default="biased")
    parser.add_argument("--condition", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--island", type=int, default=0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=16)
    parser.add_argument("--max-agents", type=int, default=120)
    parser.add_argument(
        "--total-individuals",
        type=int,
        default=None,
        help="Override population size for lighter live viewing.",
    )
    parser.add_argument(
        "--num-islands",
        type=int,
        default=None,
        help="Override island count for lighter live viewing.",
    )
    return parser.parse_args()


def _make_env(args: argparse.Namespace):
    config = make_article_task_config(
        args.task,
        variant=args.variant,
        condition=args.condition,
        seed=args.seed,
    )
    if args.total_individuals is not None:
        config["total_individuals"] = args.total_individuals
    if args.num_islands is not None:
        config["num_islands"] = args.num_islands
    if args.task == "clamity":
        return ArticleClamityEnv(config)
    return ArticleAllelopathyEnv(config)


def _agents_on_island(env: Any, island: int) -> list[tuple[str, Any]]:
    agents = [
        (agent_id, state)
        for agent_id, state in env.agent_states.items()
        if state.island == island
    ]
    return sorted(agents, key=lambda item: (item[1].species, item[0]))


def _draw_label(screen: pygame.Surface, font: pygame.font.Font, text: str, x: int, y: int) -> None:
    surface = font.render(text, True, TEXT)
    screen.blit(surface, (x, y))


def _draw_allelopathy(
    screen: pygame.Surface,
    env: ArticleAllelopathyEnv,
    island: int,
    cell: int,
    top: int,
    left: int,
) -> None:
    grid = env.shrubs.get(island)
    if grid is None:
        return
    for row in range(env.height):
        for col in range(env.width):
            value = int(grid[row, col])
            rect = pygame.Rect(left + col * cell, top + row * cell, cell, cell)
            pygame.draw.rect(screen, SHRUB_COLORS.get(value, SHRUB_COLORS[-1]), rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)


def _draw_clamity(
    screen: pygame.Surface,
    env: ArticleClamityEnv,
    island: int,
    cell: int,
    top: int,
    left: int,
) -> None:
    for row in range(env.height):
        for col in range(env.width):
            rect = pygame.Rect(left + col * cell, top + row * cell, cell, cell)
            pygame.draw.rect(screen, BACKGROUND, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)
    for row, col in env.nutrient_patches:
        rect = pygame.Rect(left + int(col) * cell, top + int(row) * cell, cell, cell)
        pygame.draw.rect(screen, NUTRIENT, rect)

    for _, state in _agents_on_island(env, island):
        if not state.settled:
            continue
        radius_px = max(cell // 2, int((state.shell_radius + 0.5) * cell))
        center = (
            left + state.col * cell + cell // 2,
            top + state.row * cell + cell // 2,
        )
        pygame.draw.circle(screen, SHELL, center, radius_px, width=max(1, cell // 5))


def _draw_agents(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: Any,
    island: int,
    cell: int,
    top: int,
    left: int,
    max_agents: int,
) -> int:
    visible_agents = _agents_on_island(env, island)[:max_agents]
    radius = max(3, cell // 3)
    for agent_id, state in visible_agents:
        color = SPECIES_COLORS[state.species % len(SPECIES_COLORS)]
        center = (
            left + state.col * cell + cell // 2,
            top + state.row * cell + cell // 2,
        )
        pygame.draw.circle(screen, color, center, radius)
        pygame.draw.circle(screen, (25, 25, 25), center, radius, width=1)
        if getattr(state, "settled", False):
            pygame.draw.circle(screen, SETTLED_CENTER, center, max(2, radius // 2))
    if len(_agents_on_island(env, island)) > max_agents:
        _draw_label(
            screen,
            font,
            f"showing {max_agents}/{len(_agents_on_island(env, island))} agents",
            left,
            top + env.height * cell + 8,
        )
    return len(visible_agents)


def _draw_sidebar(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env: Any,
    args: argparse.Namespace,
    island: int,
    x: int,
) -> None:
    agents = _agents_on_island(env, island)
    counts: dict[int, int] = {}
    returns: dict[int, list[float]] = {}
    for _, state in agents:
        counts[state.species] = counts.get(state.species, 0) + 1
        returns.setdefault(state.species, []).append(float(state.cumulative_reward))

    lines = [
        f"task: {args.task}",
        f"variant: {args.variant}",
        f"condition: {args.condition or 'default'}",
        f"island: {island}",
        f"step: {env.current_step}/{env.episode_horizon}",
        f"agents: {len(agents)}",
        "",
        "species counts",
    ]
    for species in sorted(counts):
        mean_return = sum(returns[species]) / max(1, len(returns[species]))
        lines.append(f"  s{species}: {counts[species]}  r={mean_return:.1f}")

    if hasattr(env, "shrubs"):
        grid = env.shrubs.get(island)
        if grid is not None:
            lines.extend([
                "",
                "shrubs",
                f"  A: {int((grid == 0).sum())}",
                f"  B: {int((grid == 1).sum())}",
            ])
    else:
        settled = sum(1 for _, state in agents if state.settled)
        mean_shell = (
            sum(state.shell_radius for _, state in agents) / max(1, len(agents))
        )
        lines.extend([
            "",
            "clamity",
            f"  settled: {settled}",
            f"  mean shell: {mean_shell:.2f}",
        ])

    lines.extend(["", "space pause", "right step", "esc quit"])
    for idx, line in enumerate(lines):
        _draw_label(screen, font, line, x, 24 + idx * 22)


def _random_actions(env: Any) -> dict[str, int]:
    return {
        agent_id: env.action_spaces[agent_id].sample()
        for agent_id in env.agents
    }


def main() -> None:
    args = _parse_args()
    env = _make_env(args)
    env.reset(seed=args.seed)
    island = min(max(0, args.island), env.total_island_count - 1)

    pygame.init()
    font = pygame.font.SysFont(None, 22)
    cell = max(6, args.cell_size)
    margin = 18
    sidebar_width = 280
    width = margin * 3 + env.width * cell + sidebar_width
    height = max(margin * 2 + env.height * cell + 36, 520)
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Article {args.task} viewer")
    clock = pygame.time.Clock()

    paused = False
    running = True
    while running:
        step_once = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    step_once = True
                elif event.key == pygame.K_LEFT:
                    island = (island - 1) % env.total_island_count
                elif event.key == pygame.K_TAB:
                    island = (island + 1) % env.total_island_count

        if not paused or step_once:
            _, _, _, truncations, _ = env.step(_random_actions(env))
            if truncations.get("__all__", False):
                env.reset()

        screen.fill((232, 229, 218))
        grid_top = margin
        grid_left = margin
        if isinstance(env, ArticleAllelopathyEnv):
            _draw_allelopathy(screen, env, island, cell, grid_top, grid_left)
        else:
            _draw_clamity(screen, env, island, cell, grid_top, grid_left)
        _draw_agents(screen, font, env, island, cell, grid_top, grid_left, args.max_agents)
        _draw_sidebar(
            screen,
            font,
            env,
            args,
            island,
            margin * 2 + env.width * cell,
        )

        if math.isfinite(args.fps) and args.fps > 0:
            clock.tick(args.fps)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
