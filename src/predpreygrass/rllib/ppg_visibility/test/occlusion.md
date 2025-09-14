# Line-of-Sight Occlusion for PredPreyGrass (v2_7)

Observer: predator at window center (odd obs_range). Channels:
0=walls, 1=predators, 2=prey, 3=grass, (optional 4=visibility).

Rules:
- Use LoS with Bresenham-style rays precomputed per (dy,dx) for the obs window.
- When building an observer’s patch, apply a visibility mask to **all** channels:
  - Cells with vis=0 are set to 0 (or sentinel), including prey/predator/grass.
  - Walls themselves are visible (they block further cells along the ray).
- Optionally append a binary visibility channel (1=visible, 0=occluded).
- Our worked example: 7×7, predator at (4,4), walls at (3,5..7),
  prey at (2,2) (visible) and (2,6) (occluded because ray hits (3,5)).

Targets:
- Add `self.walls: (H,W) bool`.
- Precompute `los_table[(dy,dx)] -> [(ry,rx), ...]` once per obs_range.
- Implement `extract_patch_with_occlusion(...)`.
- Update observation_space (+1 channel if using visibility).
