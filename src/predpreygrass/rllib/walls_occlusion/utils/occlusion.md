## Occlusion & Line-of-Sight Protocol (Current Spec)

Version context: Original note (v2_7) kept below; this section reflects the updated predator‑centric tooling (`print_occlusion.py`).

### 1. Entities & Channels
Typical stacked observation channels (local window, odd side length, predator centered):
0 = walls (binary)
1 = predators
2 = prey
3 = grass
4 = (optional) visibility mask (1 = visible, 0 = occluded) – only if enabled

The predator observer is always placed at the geometric center of the local window (for size `k = 2r+1`, center index = `(r, r)`). We denote the observing predator with `P` in ASCII debug output, walls with `#`, visible free cells with `.` and occluded free cells as space.

### 2. Visibility (LoS) Rule
Let predator position be `P`, target cell `T`, wall set `W`.

`T` is visible from `P` iff either:
1. `P == T`, OR
2. No wall cell lies strictly between `P` and `T` along the discrete Bresenham line connecting them.

Algorithm:
```
path = bresenham_line(P, T)          # includes endpoints P and T
internal = path[1:-1]                # exclude endpoints
visible = all(cell not in W for cell in internal)
```

Walls occupy their own cells but are themselves visible if the line is clear up to them (endpoints don’t block). They DO block any further cells along that same line.

### 3. Bresenham Decision Rule (Symmetric Form)
Internal algorithm used:
```
dx = abs(x1 - x0)
dy = -abs(y1 - y0)
sx = 1 if x0 < x1 else -1
sy = 1 if y0 < y1 else -1
err = dx + dy
while True:
    plot(x, y)
    if x == x1 and y == y1: break
    e2 = 2 * err
    if e2 >= dy: err += dy; x += sx
    if e2 <= dx: err += dx; y += sy
```
This unified form covers all octants without separate steep/shallow branches.

### 4. Predator Origin & Wall Relocation
If the chosen predator origin cell is a wall and `--strict-origin` is NOT passed:
1. That wall is removed from `W`.
2. The system searches outward (in square rings) for the nearest free cell to relocate the wall.
3. If found, it is inserted there; otherwise the wall is dropped (warning emitted).

`--strict-origin` forces an error instead (no relocation).

### 5. Script (`print_occlusion.py`) Usage
Core options:
`--width W --height H`   : Grid dimensions (global / toy grid).
`--walls x,y ...`        : Space‑separated coordinates of wall cells.
`--origin x,y`           : Explicit predator position.
`--center`               : Auto set predator to `(W//2, H//2)` if `--origin` omitted.
`--origin-matrix`        : Print wall channel + predator visibility channel (spatial 2D).
`--full-matrix`          : (Opt‑in) print full N×N pairwise visibility matrix (large!).
`--strict-origin`        : Do not auto-relocate wall at predator origin.

Example (7×7, centered predator, show channels):
```
python -m predpreygrass.rllib.visibility.print_occlusion \
  --width 7 --height 7 \
  --walls 2,2 3,3 4,1 5,5 \
  --center --origin-matrix
```

Output legend:
`P` = predator (observer)  `#` = wall  `.` = visible free cell  (space) = blocked free cell
Numeric matrix view uses `1` (visible) / `0` (blocked) plus `P`, `#` overlays.

### 6. Applying Visibility Mask to Observation
For each local window cell `(i,j)`:
```
if visibility_mask[i,j] == 0:
    walls[i,j] = 0          # (or retain 1 if you want walls always perceivable)
    predators[i,j] = 0
    prey[i,j] = 0
    grass[i,j] = 0
    # Optional: leave visibility channel as 0
```
Current spec keeps walls visible if *on* the ray up to their position; downstream code can choose to zero them after blocking if preferred.

### 7. Potential Variants (Future Extensions)
| Variant | Purpose | Change |
|---------|---------|--------|
| Supercover LoS | Eliminate corner peeking | Consider all grid squares touched by ideal line (both cells when diagonally crossing a corner) |
| Radius limit | Sensory range constraint | Enforce `if max(|dx|,|dy|) > R: occluded` |
| Endpoint wall block | Treat walls as opaque targets | If `T in W`: visible = False |
| Partial transparency | Soft vision through foliage | Accumulate opacity, block if sum > threshold |
| Directional cone | FOV / heading constraints | Filter by angle relative to predator orientation |

### 8. Edge Cases & Clarifications
| Situation | Result | Notes |
|-----------|--------|-------|
| `P == T` | Visible | Reflexive base case |
| Target wall cell | Visible if no intermediate wall | Endpoint not tested; can be toggled |
| Diagonal between tight walls | Visible (current) | Due to single Bresenham path; supercover would block |
| Predator origin initially wall | Wall relocated (default) | Error only with `--strict-origin` |
| All surrounding cells walls | Relocation may fail | Wall dropped, warning emitted |

### 9. Performance Notes
For a full matrix (if enabled): complexity ≈ `O(N * max(width, height))` where `N = width * height`. The default workflow now **avoids** printing the full matrix unless `--full-matrix` is specified.

### 10. Minimal Pseudocode for Integration
```
def compute_visibility(P, walls, targets):
    vis = set()
    for T in targets:
        if P == T: vis.add(T); continue
        line = bresenham_line(P, T)
        blocked = any(c in walls for c in line[1:-1])
        if not blocked:
            vis.add(T)
    return vis
```

### 11. Backward Compatibility (v2_7 Notes Retained)
Original intent (v2_7) included precomputing `los_table[(dy,dx)]` and applying a mask over all feature channels. That still aligns with the current protocol; only the *debug tooling and symbols* have evolved (using `P` instead of implicit center, optional relocation, opt‑in full matrix).

### 12. Changelog
| Date | Change |
|------|--------|
| 2025-09-13 | Added predator-centric notation (`P`), wall relocation rule, opt‑in `--full-matrix`, clarified Bresenham logic, future variant table. |

---
Questions or extension requests: see `print_occlusion.py` or open an issue referencing this spec.
