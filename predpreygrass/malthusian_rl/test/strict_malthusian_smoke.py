"""Ray-free smoke test for strict Malthusian mode.

This module validates the environment-level strict replication path without
spawning Ray workers, which makes it useful for CI and local smoke checks.
"""

from predpreygrass.malthusian_rl.config.config_env import config_env
from predpreygrass.malthusian_rl.predpreygrass_rllib_env import PredPreyGrass


def run_smoke_validation() -> dict:
    env_config = dict(config_env)
    env_config["max_steps"] = 5
    env = PredPreyGrass(env_config)

    observations, _ = env.reset(seed=123)
    if not observations:
        raise RuntimeError("Expected initial observations from reset().")

    if env.malthusian_replication_mode != "strict":
        raise RuntimeError(f"Expected strict mode, got {env.malthusian_replication_mode!r}")
    if env.malthusian_mu_update != "multiplicative":
        raise RuntimeError(f"Expected multiplicative mu update, got {env.malthusian_mu_update!r}")
    if env.enable_within_episode_reproduction:
        raise RuntimeError("Expected within-episode reproduction to be disabled in strict mode.")

    mu_before = {species: dict(mu) for species, mu in env.mu_by_species.items()}

    while env.current_step < env.max_steps:
        action_dict = {agent_id: env.action_spaces[agent_id].sample() for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(action_dict)

    observations, rewards, terminations, truncations, infos = env.step({})
    if not truncations.get("__all__"):
        raise RuntimeError("Expected truncation at episode boundary.")
    summary = infos.get("__all__", {})
    if summary.get("malthusian_replication_mode") != "strict":
        raise RuntimeError("Episode summary did not report strict replication mode.")
    if summary.get("malthusian_mu_update") != "multiplicative":
        raise RuntimeError("Episode summary did not report multiplicative mu update.")
    if "phi_by_species" not in summary:
        raise RuntimeError("Episode summary did not include phi_by_species.")

    mu_after = summary.get("mu_by_species", {})
    return {
        "strict_mode": env.malthusian_replication_mode,
        "mu_update": env.malthusian_mu_update,
        "within_episode_reproduction": env.enable_within_episode_reproduction,
        "islands": len(env.island_id_to_cells),
        "mu_before": mu_before,
        "mu_after": mu_after,
        "phi_by_species": summary.get("phi_by_species", {}),
    }


def main() -> None:
    result = run_smoke_validation()
    print("strict_mode", result["strict_mode"])
    print("mu_update", result["mu_update"])
    print("within_episode_reproduction", result["within_episode_reproduction"])
    print("islands", result["islands"])
    print("mu_before", result["mu_before"])
    print("mu_after", result["mu_after"])
    print("phi_by_species", result["phi_by_species"])


if __name__ == "__main__":
    main()