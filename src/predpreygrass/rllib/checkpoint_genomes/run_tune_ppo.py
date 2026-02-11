"""
Run tune_ppo.py using defaults from tune_ppo_config.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _resolve(path_str: str, base_dir: Optional[Path]) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is not None:
        candidate = (base_dir / path).expanduser().resolve()
        return candidate
    return (Path.cwd() / path).expanduser().resolve()


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def main():
    config_path = Path(__file__).with_name("tune_ppo_config.json")
    cfg = _load_config(config_path)
    base_dir = config_path.parent

    restore_checkpoint = cfg.get("restore_checkpoint")
    trained_example_dir = cfg.get("trained_example_dir")

    env = os.environ.copy()
    if restore_checkpoint:
        env["RESTORE_CHECKPOINT"] = str(_resolve(restore_checkpoint, base_dir))
    if trained_example_dir:
        env["TRAINED_EXAMPLE_DIR"] = str(_resolve(trained_example_dir, base_dir))

    tune_ppo_path = _resolve("tune_ppo.py", base_dir)
    subprocess.run([sys.executable, str(tune_ppo_path)], check=True, env=env)


if __name__ == "__main__":
    main()
