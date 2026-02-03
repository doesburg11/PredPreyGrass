#!/usr/bin/env python3
"""Generate a grid-style SVG diagram for forward-shifted predator observations."""

from __future__ import annotations

import base64
from pathlib import Path


GRID_SIZE = 9
OBS_RANGE = 5
OFFSET = (OBS_RANGE - 1) // 2
PRED_POS = (4, 4)
CARDINAL_MOVE = (0, 1)  # dx, dy (up)
DIAGONAL_MOVE = (1, 1)  # dx, dy (up-right)
CELL = 32
PADDING = 24
TITLE_H = 32
PANEL_GAP = 24


def _cell_center(origin_x: int, origin_y: int, x: int, y: int) -> tuple[float, float]:
    cx = origin_x + x * CELL + CELL / 2.0
    cy = origin_y + (GRID_SIZE - 1 - y) * CELL + CELL / 2.0
    return cx, cy


def _rect_top_left(origin_x: int, origin_y: int, x: int, y: int) -> tuple[int, int]:
    rx = origin_x + x * CELL
    ry = origin_y + (GRID_SIZE - 1 - y) * CELL
    return rx, ry


def _obs_rect_bounds(center: tuple[int, int]) -> tuple[int, int, int, int]:
    xlo = center[0] - OFFSET
    xhi = center[0] + OFFSET
    ylo = center[1] - OFFSET
    yhi = center[1] + OFFSET
    return xlo, xhi, ylo, yhi


def _draw_grid(lines: list[str], origin_x: int, origin_y: int) -> None:
    width = GRID_SIZE * CELL
    height = GRID_SIZE * CELL
    for i in range(GRID_SIZE + 1):
        x = origin_x + i * CELL
        lines.append(
            f'<line x1="{x}" y1="{origin_y}" x2="{x}" y2="{origin_y + height}" '
            'stroke="#b0b0b0" stroke-width="1" stroke-dasharray="3,3" />'
        )
    for j in range(GRID_SIZE + 1):
        y = origin_y + j * CELL
        lines.append(
            f'<line x1="{origin_x}" y1="{y}" x2="{origin_x + width}" y2="{y}" '
            'stroke="#b0b0b0" stroke-width="1" stroke-dasharray="3,3" />'
        )


def _encode_icon(path: Path) -> str | None:
    if not path.is_file():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return encoded


def _draw_icon(lines: list[str], origin_x: int, origin_y: int, pos: tuple[int, int], data_uri: str | None) -> None:
    cx, cy = _cell_center(origin_x, origin_y, *pos)
    size = CELL * 0.9
    x = cx - size / 2
    y = cy - size / 2
    if data_uri:
        lines.append(
            f'<image x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" '
            f'href="{data_uri}" />'
        )
        return
    lines.append(
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{CELL * 0.28:.1f}" '
        'fill="#c22f2f" stroke="#000000" stroke-width="1" />'
    )


def _draw_obs_window(lines: list[str], origin_x: int, origin_y: int, center: tuple[int, int]) -> None:
    xlo, xhi, ylo, yhi = _obs_rect_bounds(center)
    rx, ry = _rect_top_left(origin_x, origin_y, xlo, yhi)
    width = (xhi - xlo + 1) * CELL
    height = (yhi - ylo + 1) * CELL
    lines.append(
        f'<rect x="{rx}" y="{ry}" width="{width}" height="{height}" '
        'fill="#2f6f9f" fill-opacity="0.15" stroke="#2f6f9f" stroke-width="2" />'
    )


def _draw_intended_move(lines: list[str], origin_x: int, origin_y: int, move: tuple[int, int]) -> None:
    dx, dy = move
    cx, cy = _cell_center(origin_x, origin_y, *PRED_POS)
    end_x = cx + dx * CELL * 0.8
    end_y = cy - dy * CELL * 0.8
    lines.append(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" '
        'stroke="#000000" stroke-width="2" marker-end="url(#arrowhead)" />'
    )
    label_x = end_x + dx * 8
    label_y = end_y - dy * 8
    lines.append(
        f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="12" fill="#000000">intended move</text>'
    )


def _line_of_sight_clear(start: tuple[int, int], end: tuple[int, int], walls: set[tuple[int, int]]) -> bool:
    (x0, y0), (x1, y1) = start, end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    if dx >= dy:
        err = dx / 2.0
        while x != x1:
            if (x, y) not in (start, end) and (x, y) in walls:
                return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            if (x, y) not in (start, end) and (x, y) in walls:
                return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return True


def _draw_los_mask(
    lines: list[str],
    origin_x: int,
    origin_y: int,
    obs_center: tuple[int, int],
    agent_pos: tuple[int, int],
    walls: set[tuple[int, int]],
) -> None:
    xlo, xhi, ylo, yhi = _obs_rect_bounds(obs_center)
    for gx in range(xlo, xhi + 1):
        for gy in range(ylo, yhi + 1):
            if _line_of_sight_clear(agent_pos, (gx, gy), walls):
                rx, ry = _rect_top_left(origin_x, origin_y, gx, gy)
                lines.append(
                    f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" '
                    'fill="#f2c94c" fill-opacity="0.18" stroke="none" />'
                )


def _panel(
    lines: list[str],
    origin_x: int,
    origin_y: int,
    title: str,
    center: tuple[int, int],
    *,
    show_arrow: bool,
    move: tuple[int, int] | None,
    icon_uri: str | None,
    walls: set[tuple[int, int]] | None = None,
    show_los: bool = False,
) -> None:
    lines.append(
        f'<text x="{origin_x}" y="{origin_y - 10}" font-size="12" fill="#000000">{title}</text>'
    )
    _draw_grid(lines, origin_x, origin_y)
    _draw_obs_window(lines, origin_x, origin_y, center)
    if show_los and walls:
        _draw_los_mask(lines, origin_x, origin_y, center, PRED_POS, walls)
    if walls:
        for wx, wy in walls:
            rx, ry = _rect_top_left(origin_x, origin_y, wx, wy)
            lines.append(
                f'<rect x="{rx}" y="{ry}" width="{CELL}" height="{CELL}" '
                'fill="#3b3b3b" stroke="#222222" stroke-width="1" />'
            )
    _draw_icon(lines, origin_x, origin_y, PRED_POS, icon_uri)
    if show_arrow and move:
        _draw_intended_move(lines, origin_x, origin_y, move)
    if show_los and walls:
        lines.append(
            f'<text x="{origin_x}" y="{origin_y + GRID_SIZE * CELL + 16}" font-size="11" fill="#555555">'
            "LOS mask shown (visible cells)</text>"
        )


def main() -> Path:
    px, py = PRED_POS
    centered_center = (px, py)
    forward_center_cardinal = (px + CARDINAL_MOVE[0] * OFFSET, py + CARDINAL_MOVE[1] * OFFSET)
    forward_center_diag = (px + DIAGONAL_MOVE[0] * OFFSET, py + DIAGONAL_MOVE[1] * OFFSET)

    panel_w = GRID_SIZE * CELL
    panel_h = GRID_SIZE * CELL
    width = PADDING * 2 + panel_w * 3 + PANEL_GAP * 2
    height = PADDING * 2 + panel_h + TITLE_H
    origin_y = PADDING + TITLE_H
    left_x = PADDING
    middle_x = PADDING + panel_w + PANEL_GAP
    right_x = middle_x + panel_w + PANEL_GAP

    icons_dir = Path(__file__).resolve().parents[5] / "assets" / "images" / "icons"
    predator_icon = _encode_icon(icons_dir / "human_1.png")
    prey_icon = _encode_icon(icons_dir / "prey.png")
    predator_uri = f"data:image/png;base64,{predator_icon}" if predator_icon else None
    prey_uri = f"data:image/png;base64,{prey_icon}" if prey_icon else None

    walls = {
        (6, 5),
        (6, 6),
        (6, 7),
        (7, 6),
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<defs>',
        '<marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
        '<polygon points="0 0, 8 3, 0 6" fill="#000000" />'
        "</marker>",
        "</defs>",
        '<style>text { font-family: sans-serif; }</style>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{PADDING}" y="{PADDING}" font-size="14" fill="#000000">'
        "Predator observation shifts forward by OFFSET = (obs_range - 1) / 2</text>",
    ]

    _panel(
        lines,
        left_x,
        origin_y,
        "Centered observation (prey)",
        centered_center,
        show_arrow=False,
        move=None,
        icon_uri=prey_uri,
    )
    _panel(
        lines,
        middle_x,
        origin_y,
        "Forward-shifted (human, cardinal)",
        forward_center_cardinal,
        show_arrow=True,
        move=CARDINAL_MOVE,
        icon_uri=predator_uri,
    )
    _panel(
        lines,
        right_x,
        origin_y,
        "Forward-shifted (human, diagonal) + LOS",
        forward_center_diag,
        show_arrow=True,
        move=DIAGONAL_MOVE,
        icon_uri=predator_uri,
        walls=walls,
        show_los=True,
    )

    lines.append("</svg>")

    out_dir = Path(__file__).resolve().parents[5] / "assets" / "images" / "readme"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forward_view_observation_shift.svg"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Saved diagram to {path}")
