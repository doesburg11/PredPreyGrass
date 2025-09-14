"""Utility: Print an occlusion (line-of-sight visibility) matrix for a grid.

Each cell is considered *visible* from another cell if the straight line segment
between their centers does not pass through any *wall* cell (excluding the
endpoints). We use an integer grid Bresenham traversal to test for blocking.

The resulting matrix M has shape (W*H, W*H) with:
  M[i, j] = 1  if cell j is visible from cell i (including i == j)
          = 0  otherwise.

Cell linear index convention: idx = y * width + x (row-major order).

Example:
  python -m predpreygrass.rllib.visibility.print_occlusion \\
      --width 5 --height 4 --walls 1,1 2,1 3,2

  # Only print matrix for a single origin cell (2,2):
  python -m predpreygrass.rllib.visibility.print_occlusion \\
      --width 5 --height 4 --walls 1,1 2,1 3,2 --origin 2,2

Notes:
  * Diagonal visibility is allowed if no wall cell lies exactly on the Bresenham
    path between the two cells.
  * This script is standalone and does not modify environment logic; it's a
    debugging / design aid for occlusion specifications documented in
    `occlusion.md`.
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence, Set, Tuple


Coord = Tuple[int, int]


def parse_walls(values: Sequence[str]) -> List[Coord]:
    walls = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        try:
            x_str, y_str = v.split(",")
            x, y = int(x_str), int(y_str)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Wall '{v}' must be in 'x,y' integer format"
            ) from e
        walls.append((x, y))
    return walls


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Coord]:
    """Return list of grid coordinates along line from (x0,y0) to (x1,y1),
    inclusive, using Bresenham's algorithm.

    We will later exclude the first and last point (the endpoints) when testing
    for blockers.
    """
    points: List[Coord] = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def visible(a: Coord, b: Coord, wall_set: Set[Coord]) -> bool:
    if a == b:
        return True
    pts = bresenham_line(a[0], a[1], b[0], b[1])
    # Exclude endpoints (a and b). Any wall strictly between blocks.
    for p in pts[1:-1]:
        if p in wall_set:
            return False
    return True


def build_visibility_matrix(width: int, height: int, walls: Iterable[Coord]):
    wall_set = set(walls)
    n = width * height
    # Use list of bytearray for memory efficiency
    matrix = [bytearray(n) for _ in range(n)]
    coords: List[Coord] = [(x, y) for y in range(height) for x in range(width)]

    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            matrix[i][j] = 1 if visible(a, b, wall_set) else 0
    return matrix, coords


def format_matrix(matrix: List[bytearray]) -> str:
    lines = []
    for row in matrix:
        lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines)


def print_origin_visibility(origin: Coord, width: int, height: int, walls: Set[Coord]):
    """Pretty-print a visibility grid from a single origin (predator) cell.

    Legend:
      P = predator (observer)
      # = wall
      . = visible cell (line-of-sight clear)
        = blocked cell
    """
    ox, oy = origin
    lines = []
    for y in range(height):
        row_chars = []
        for x in range(width):
            if (x, y) == origin:
                row_chars.append("P")  # Predator origin
            elif (x, y) in walls:
                row_chars.append("#")  # Wall
            else:
                row_chars.append("." if visible(origin, (x, y), walls) else " ")
        lines.append("".join(row_chars))
    print("Predator visibility (P = predator, # = wall, '.' = visible, space = blocked):")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Print occlusion (visibility) matrix for a grid with wall cells.")
    parser.add_argument("--width", type=int, required=True, help="Grid width")
    parser.add_argument("--height", type=int, required=True, help="Grid height")
    parser.add_argument(
        "--walls",
        nargs="*",
        default=[],
        help="List of wall cells as 'x,y' (space separated). Example: --walls 1,1 2,3 4,0",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help="Optional single origin 'x,y' to print a 2D visibility map instead of full matrix.",
    )
    parser.add_argument(
        "--origin-matrix",
        action="store_true",
        help="If set with --origin, also print a numeric 2D matrix (H x W) with O/#/1/0 inline (walls shown inline, not in a separate list).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Automatically set predator origin to the geometric center (width//2,height//2). Ignored if --origin is provided.",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Opt-in: print the full N x N visibility matrix (can be very large). Default is to skip it.",
    )
    parser.add_argument(
        "--strict-origin",
        action="store_true",
        help="If set, raise an error when the predator origin cell is a wall instead of relocating that wall.",
    )
    args = parser.parse_args()

    walls = parse_walls(args.walls)
    wall_set = set(walls)
    matrix, coords = build_visibility_matrix(args.width, args.height, walls)

    # Auto-center if requested and no explicit origin given.
    if not args.origin and args.center:
        args.origin = f"{args.width//2},{args.height//2}"

    if args.origin:
        try:
            ox_str, oy_str = args.origin.split(",")
            origin = (int(ox_str), int(oy_str))
        except Exception as e:
            raise SystemExit(f"--origin must be 'x,y': {e}")
        if not (0 <= origin[0] < args.width and 0 <= origin[1] < args.height):
            raise SystemExit("Origin out of bounds")
        if origin in wall_set:
            if args.strict_origin:
                raise SystemExit("Invalid configuration: predator origin cannot occupy a wall cell (strict mode).")
            # Relocate the wall to the nearest free cell.
            original_origin = origin
            # Remove the wall at origin.
            wall_set.remove(origin)
            # Search outward for nearest free cell to place the relocated wall.
            found_spot = None
            max_r = max(args.width, args.height)
            for r in range(1, max_r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            # Only check perimeter of the square ring for efficiency
                            continue
                        nx = origin[0] + dx
                        ny = origin[1] + dy
                        if 0 <= nx < args.width and 0 <= ny < args.height:
                            cand = (nx, ny)
                            if cand not in wall_set and cand != origin:
                                found_spot = cand
                                break
                    if found_spot:
                        break
                if found_spot:
                    break
            if not found_spot:
                print("Warning: Could not find relocation spot for wall at origin; dropping wall.")
            else:
                wall_set.add(found_spot)
                print(f"Relocated wall from predator origin {original_origin} to {found_spot}.")
        print_origin_visibility(origin, args.width, args.height, wall_set)
        if args.origin_matrix:
            # Two-channel style: first show walls, then visibility from origin.
            ox, oy = origin
            header = "    " + " ".join(f"{x:2d}" for x in range(args.width))
            # Wall channel
            print("\nWall channel (rows=y, cols=x):")
            print(header)
            for y in range(args.height):
                row_cells = []
                for x in range(args.width):
                    row_cells.append(" #" if (x, y) in wall_set else " .")
                print(f"y={y:2d} " + " ".join(row_cells))

            # Visibility channel (7x7 style if small) with O origin, # walls, 1/0 vis of free cells.
            print("\nOcclusion visibility (from predator) matrix (same spatial shape). Legend: P=predator, #=wall, 1=visible, 0=blocked")
            print(header)
            for y in range(args.height):
                row_cells = []
                for x in range(args.width):
                    if (x, y) == origin:
                        row_cells.append(" P")
                    elif (x, y) in wall_set:
                        row_cells.append(" #")
                    else:
                        row_cells.append(" 1" if visible(origin, (x, y), wall_set) else " 0")
                print(f"y={y:2d} " + " ".join(row_cells))
        print()

    if args.full_matrix:
        print(f"Full visibility (occlusion) matrix (size {len(coords)} x {len(coords)}). Row=source index, Col=target index.")
        lines = []
        for i, row in enumerate(matrix):
            line_parts = []
            ax, ay = coords[i]
            a_is_wall = (ax, ay) in wall_set
            for j, val in enumerate(row):
                if i == j and a_is_wall:
                    line_parts.append('W')
                else:
                    line_parts.append(str(val))
            lines.append(" ".join(line_parts))
        if args.width * args.height <= 49:
            print("Coordinate index grid (idx in each cell):")
            idx_grid_lines = []
            for y in range(args.height):
                idx_grid_lines.append(" ".join(f"{(y*args.width + x):2d}" for x in range(args.width)))
            print("\n".join(idx_grid_lines))
            print("(Cells with walls have 'W' on their diagonal position in matrix)\n")
        print("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover
    main()
