"""
Run make_child_checkpoint.py using defaults from make_child_checkpoint_config.json.
"""
import json
from pathlib import Path
from typing import Optional

from predpreygrass.rllib.checkpoint_genomes.make_child_checkpoint import make_child_checkpoint


def _resolve(path_str: str, base_dir: Optional[Path]) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        return (base_dir / path).expanduser().resolve()
    return (Path.cwd() / path).expanduser().resolve()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def main():
    config_path = Path(__file__).with_name("make_child_checkpoint_config.json")
    cfg = _load_config(config_path)
    base_dir = config_path.parent

    parent_a = cfg.get("parent_a")
    parent_b = cfg.get("parent_b")
    out_dir = cfg.get("out_dir")
    if not parent_a or not parent_b or not out_dir:
        raise ValueError("parent_a, parent_b, and out_dir must be set in the config.")

    alpha = float(cfg.get("alpha", 0.5))
    sigma = float(cfg.get("sigma", 0.01))
    seed = cfg.get("seed", None)
    policies = cfg.get("policies", None)
    overwrite = bool(cfg.get("overwrite", False))

    make_child_checkpoint(
        _resolve(parent_a, base_dir),
        _resolve(parent_b, base_dir),
        _resolve(out_dir, base_dir),
        alpha=alpha,
        sigma=sigma,
        seed=seed,
        policies=policies,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
