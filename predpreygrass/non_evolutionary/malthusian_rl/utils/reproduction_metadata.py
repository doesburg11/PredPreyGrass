"""Run metadata helpers for exact reproduction audits."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGE_NAMES = [
    "ray",
    "torch",
    "numpy",
    "gymnasium",
    "matplotlib",
    "pettingzoo",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_dirty_status() -> str | None:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def config_checksum(*configs: Any) -> str:
    payload = json.dumps(configs, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": package_versions(),
    }


def build_run_metadata(env_config: dict[str, Any], appo_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata_schema": "leibo_2019_reproduction_run_v1",
        "git_commit": git_commit(),
        "git_status_short": git_dirty_status(),
        "environment_snapshot": environment_snapshot(),
        "config_checksum_sha256": config_checksum(env_config, appo_config),
        "config_checksum_inputs": ["config_env", "config_appo_exact"],
    }


def verify_run_config_metadata(run_config: dict[str, Any]) -> dict[str, Any]:
    env_config = run_config.get("config_env")
    appo_config = run_config.get("config_appo_exact")
    stored_checksum = run_config.get("config_checksum_sha256")
    recomputed_checksum = (
        config_checksum(env_config, appo_config)
        if isinstance(env_config, dict) and isinstance(appo_config, dict)
        else None
    )
    checksum_valid = (
        bool(stored_checksum)
        and bool(recomputed_checksum)
        and stored_checksum == recomputed_checksum
    )
    environment_snapshot = run_config.get("environment_snapshot")
    package_versions_present = (
        isinstance(environment_snapshot, dict)
        and isinstance(environment_snapshot.get("package_versions"), dict)
    )
    git_commit_present = bool(run_config.get("git_commit"))
    return {
        "checksum_present": bool(stored_checksum),
        "checksum_valid": checksum_valid,
        "stored_checksum": stored_checksum,
        "recomputed_checksum": recomputed_checksum,
        "environment_snapshot_present": isinstance(environment_snapshot, dict),
        "package_versions_present": package_versions_present,
        "git_commit_present": git_commit_present,
        "metadata_schema": run_config.get("metadata_schema"),
    }
