"""Fixed-trait environments, with compatibility for saved discrete checkpoints."""
from pathlib import Path

# Older pickles and archived evaluators refer to non_evolutionary.<environment>.
# Search the relocated directory after this package's own directory so the
# continuous environment keeps its original resolution.
__path__.append(str(Path(__file__).resolve().parent / "discrete_space"))
