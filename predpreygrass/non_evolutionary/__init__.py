"""Fixed-trait environments, with compatibility for saved checkpoints."""
from pathlib import Path

# Older pickles and archived evaluators use paths predating the space grouping.
# Keep those module names importable after relocating the environments.
for _space in ("discrete_space", "continuous_space"):
    __path__.append(str(Path(__file__).resolve().parent / _space))
