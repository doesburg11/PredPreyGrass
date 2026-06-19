"""
Strict APPO entrypoint for Malthusian reproduction runs.

Unlike the exploratory trainer, this script imports one locked config and
validates parity invariants before launching Tune.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import (
    DefaultAPPOTorchRLModule,
)
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.registry import register_env

from predpreygrass.malthusian_rl.config.config_appo_exact import (
    config_appo_exact,
    paper_learner_citation_map,
)
from predpreygrass.malthusian_rl.config.config_paper_protocol import (
    DEFAULT_PAPER_PROTOCOL_VARIANT,
    PAPER_PROTOCOL_VARIANTS,
    make_paper_protocol_env_config,
)
from predpreygrass.malthusian_rl.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.malthusian_rl.utils.episode_return_callback import (
    EpisodeReturn,
)
from predpreygrass.malthusian_rl.utils.networks import build_multi_module_spec
from predpreygrass.malthusian_rl.utils.reproduction_metadata import (
    build_run_metadata,
)


def env_creator(config):
    return PredPreyGrass(config)


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = str(agent_id).split("_")
    if len(parts) < 3:
        raise ValueError(f"Expected agent id like 'type_1_predator_0', got {agent_id!r}")
    type_id = parts[1]
    role = parts[2]
    return f"type_{type_id}_{role}"


def validate_exact_configs(env_config, appo_config):
    required_appo = {
        "vtrace": True,
        "vtrace_clip_rho_threshold": 1.0,
        "vtrace_clip_pg_rho_threshold": 1.0,
        "use_kl_loss": False,
        "kl_coeff": 0.0,
        "use_lstm": True,
        "lstm_cell_size": 64,
        "paper_network_architecture": True,
        "opt_type": "rmsprop",
        "decay": 0.99,
        "epsilon": 0.0001,
        "num_env_runners": 1,
        "num_envs_per_env_runner": 1,
    }
    for key, expected in required_appo.items():
        actual = appo_config.get(key)
        if actual != expected:
            raise ValueError(f"Exact APPO config requires {key}={expected!r}; got {actual!r}.")

    required_env = {
        "enable_malthusian_update": True,
        "malthusian_replication_mode": "strict",
        "malthusian_mu_update": "multiplicative",
        "enable_within_episode_reproduction": False,
        "deterministic_reset_sequence": True,
    }
    for key, expected in required_env.items():
        actual = env_config.get(key)
        if actual != expected:
            raise ValueError(f"Exact env config requires {key}={expected!r}; got {actual!r}.")
    if env_config.get("paper_protocol_name") != "leibo_2019_malthusian_protocol_mapped_to_ppg":
        raise ValueError("Exact env config must come from make_paper_protocol_env_config().")


def build_policy_spaces(sample_env):
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    if observation_spaces is None or action_spaces is None:
        raise RuntimeError("PredPreyGrass must define observation_spaces and action_spaces before training.")

    obs_by_policy = {}
    act_by_policy = {}
    for agent_id, obs_space in observation_spaces.items():
        policy_id = policy_mapping_fn(agent_id)
        if policy_id not in obs_by_policy:
            obs_by_policy[policy_id] = obs_space
            act_by_policy[policy_id] = action_spaces[agent_id]

    return obs_by_policy, act_by_policy


def build_exact_appo_config(env_config, appo_config):
    sample_env = env_creator(env_config)
    obs_by_policy, act_by_policy = build_policy_spaces(sample_env)
    del sample_env

    multi_module_spec = build_multi_module_spec(
        obs_by_policy,
        act_by_policy,
        module_class=DefaultAPPOTorchRLModule,
        use_lstm=appo_config["use_lstm"],
        max_seq_len=appo_config["max_seq_len"],
        lstm_cell_size=appo_config["lstm_cell_size"],
        paper_network_architecture=appo_config["paper_network_architecture"],
    )
    policies = {
        policy_id: (None, obs_by_policy[policy_id], act_by_policy[policy_id], {})
        for policy_id in obs_by_policy
    }

    return (
        APPOConfig()
        .environment(env="PredPreyGrass", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            train_batch_size_per_learner=appo_config["train_batch_size_per_learner"],
            gamma=appo_config["gamma"],
            lr=appo_config["lr"],
            entropy_coeff=appo_config["entropy_coeff"],
            vf_loss_coeff=appo_config["vf_loss_coeff"],
            clip_param=appo_config["clip_param"],
            opt_type=appo_config["opt_type"],
            decay=appo_config["decay"],
            momentum=appo_config["momentum"],
            epsilon=appo_config["epsilon"],
            kl_coeff=appo_config["kl_coeff"],
            kl_target=appo_config["kl_target"],
            use_kl_loss=appo_config["use_kl_loss"],
            vtrace=appo_config["vtrace"],
            vtrace_clip_rho_threshold=appo_config["vtrace_clip_rho_threshold"],
            vtrace_clip_pg_rho_threshold=appo_config["vtrace_clip_pg_rho_threshold"],
            grad_clip=appo_config["grad_clip"],
            use_circular_buffer=appo_config["use_circular_buffer"],
            simple_queue_size=appo_config["simple_queue_size"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_gpus_per_learner=appo_config["num_gpus_per_learner"],
            num_learners=appo_config["num_learners"],
        )
        .env_runners(
            num_env_runners=appo_config["num_env_runners"],
            num_envs_per_env_runner=appo_config["num_envs_per_env_runner"],
            rollout_fragment_length=appo_config["rollout_fragment_length"],
            sample_timeout_s=appo_config["sample_timeout_s"],
            num_cpus_per_env_runner=appo_config["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=appo_config["num_cpus_for_main_process"],
        )
        .debugging(seed=env_config.get("seed"))
        .callbacks(EpisodeReturn)
    )


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def build_run_inputs():
    variant = os.environ.get("EXACT_PROTOCOL_VARIANT", DEFAULT_PAPER_PROTOCOL_VARIANT)
    if variant not in PAPER_PROTOCOL_VARIANTS:
        known = ", ".join(sorted(PAPER_PROTOCOL_VARIANTS))
        raise ValueError(f"EXACT_PROTOCOL_VARIANT={variant!r} is not valid. Expected one of: {known}.")

    seed = _env_int("EXACT_SEED", 0)
    max_iters = _env_int("EXACT_MAX_ITERS", config_appo_exact["max_iters"])
    checkpoint_every = _env_int("EXACT_CHECKPOINT_EVERY", config_appo_exact["checkpoint_every"])
    env_config = make_paper_protocol_env_config(variant=variant, seed=seed)

    appo_config = dict(config_appo_exact)
    appo_config["max_iters"] = max_iters
    appo_config["checkpoint_every"] = checkpoint_every

    results_dir = Path(
        os.environ.get(
            "EXACT_RESULTS_DIR",
            "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/",
        )
    ).expanduser()

    return env_config, appo_config, results_dir


if __name__ == "__main__":
    env_config, appo_run_config, ray_results_path = build_run_inputs()
    validate_exact_configs(env_config, appo_run_config)

    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("PredPreyGrass", env_creator)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = (
        f"APPO_MALTHUSIAN_EXACT_{env_config['paper_protocol_variant']}"
        f"_seed_{env_config['seed']}_{timestamp}"
    )
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(
            {
                "config_env": env_config,
                "config_appo_exact": appo_run_config,
                "paper_learner_citation_map": paper_learner_citation_map,
                "entrypoint": __file__,
                "experiment_name": experiment_name,
                **build_run_metadata(env_config, appo_run_config),
            },
            f,
            indent=4,
        )

    appo_config = build_exact_appo_config(env_config, appo_run_config)
    tuner = Tuner(
        appo_config.algo_class,
        param_space=appo_config.to_dict(),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop={"training_iteration": appo_run_config["max_iters"]},
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=appo_run_config["checkpoint_every"],
                checkpoint_at_end=True,
            ),
        ),
    )

    tuner.fit()
    ray.shutdown()
