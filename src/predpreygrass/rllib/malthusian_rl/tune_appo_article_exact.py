"""
APPO entrypoint for article-task reconstructions.

This trainer targets `ArticleAllelopathyEnv` and `ArticleClamityEnv`, not the
Predator-Prey-Grass mapped protocol.
"""

from __future__ import annotations

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

from predpreygrass.rllib.malthusian_rl.article_tasks import (
    ArticleAllelopathyEnv,
    ArticleClamityEnv,
)
from predpreygrass.rllib.malthusian_rl.config.config_appo_exact import (
    config_appo_exact,
    paper_learner_citation_map,
)
from predpreygrass.rllib.malthusian_rl.config.config_article_protocol import (
    ARTICLE_EXACT_BLOCKERS,
    make_article_task_config,
)
from predpreygrass.rllib.malthusian_rl.utils.episode_return_callback import (
    EpisodeReturn,
)
from predpreygrass.rllib.malthusian_rl.utils.mu_server import make_mu_server
from predpreygrass.rllib.malthusian_rl.utils.networks import build_multi_module_spec
from predpreygrass.rllib.malthusian_rl.utils.reproduction_metadata import (
    build_run_metadata,
)


def article_policy_mapping_fn(agent_id, *args, **kwargs):
    del args, kwargs
    parts = str(agent_id).split("_")
    if len(parts) < 2 or parts[0] != "species":
        raise ValueError(f"Expected agent id like 'species_0_agent_0', got {agent_id!r}")
    return f"species_{parts[1]}"


def article_env_creator(config):
    task = config.get("task")
    if task == "allelopathy":
        return ArticleAllelopathyEnv(config)
    if task == "clamity":
        return ArticleClamityEnv(config)
    raise ValueError("Article env config requires task='allelopathy' or task='clamity'.")


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def build_policy_spaces(sample_env):
    observation_spaces = sample_env.observation_spaces
    action_spaces = sample_env.action_spaces
    obs_by_policy = {}
    act_by_policy = {}
    for agent_id, obs_space in observation_spaces.items():
        policy_id = article_policy_mapping_fn(agent_id)
        if policy_id not in obs_by_policy:
            obs_by_policy[policy_id] = obs_space
            act_by_policy[policy_id] = action_spaces[agent_id]
    return obs_by_policy, act_by_policy


def build_article_appo_config(env_config, appo_config, mu_server=None):
    # Build spaces from a standalone env that does NOT connect to the MuServer,
    # so the sample env does not consume an island registration slot.
    _sample_config = {k: v for k, v in env_config.items() if k != "mu_server"}
    sample_env = article_env_creator(_sample_config)
    obs_by_policy, act_by_policy = build_policy_spaces(sample_env)
    del sample_env

    if mu_server is not None:
        env_config = dict(env_config)
        env_config["mu_server"] = mu_server

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
        .environment(env="ArticleMalthusianTask", env_config=env_config)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=article_policy_mapping_fn,
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
        .resources(num_cpus_for_main_process=appo_config["num_cpus_for_main_process"])
        .debugging(seed=env_config.get("seed"))
        .callbacks(EpisodeReturn)
    )


def build_run_inputs():
    task = os.environ.get("ARTICLE_TASK", "allelopathy")
    variant = os.environ.get("ARTICLE_VARIANT", "biased")
    condition = os.environ.get("ARTICLE_CONDITION") or None
    seed = _env_int("ARTICLE_SEED", 0)
    max_iters = _env_int("ARTICLE_MAX_ITERS", config_appo_exact["max_iters"])
    checkpoint_every = _env_int("ARTICLE_CHECKPOINT_EVERY", config_appo_exact["checkpoint_every"])
    env_config = make_article_task_config(task, variant=variant, condition=condition, seed=seed or 0)
    appo_config = dict(config_appo_exact)
    appo_config["max_iters"] = max_iters
    appo_config["checkpoint_every"] = checkpoint_every

    # ARTICLE_DISTRIBUTED_RUNNERS: set to NI (e.g. 60) to enable distributed
    # island training via MuServer.  Each runner handles one island, which
    # matches the paper's multi-process island architecture (Section 2.4).
    distributed_runners = _env_int("ARTICLE_DISTRIBUTED_RUNNERS", None)
    if distributed_runners:
        appo_config["num_env_runners"] = distributed_runners
        appo_config["num_envs_per_env_runner"] = 1

    results_dir = Path(
        os.environ.get(
            "ARTICLE_RESULTS_DIR",
            "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/",
        )
    ).expanduser()
    return env_config, appo_config, results_dir, distributed_runners


if __name__ == "__main__":
    env_config, appo_run_config, ray_results_path, _distributed_runners = build_run_inputs()
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)
    register_env("ArticleMalthusianTask", article_env_creator)

    # Create a shared MuServer when running in distributed-island mode.
    # In single-process mode (_distributed_runners is None), mu_server=None and
    # all island state stays local inside the env (existing behaviour).
    mu_server = None
    if _distributed_runners:
        mu_server = make_mu_server(
            num_species=int(env_config.get("num_species", 4)),
            num_islands=int(env_config.get("num_islands", 60)),
            alpha=float(env_config.get("alpha", 0.0001)),
            eta=float(env_config.get("eta", 0.01)),
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = (
        f"APPO_MALTHUSIAN_ARTICLE_{env_config['task']}_{env_config.get('variant', 'default')}"
        f"_{env_config.get('experiment_condition', 'default')}"
        f"_seed_{env_config['seed']}_{timestamp}"
    )
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    _serializable_env_config = {k: v for k, v in env_config.items() if k != "mu_server"}
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(
            {
                "config_env": _serializable_env_config,
                "config_appo_exact": appo_run_config,
                "paper_learner_citation_map": paper_learner_citation_map,
                "article_exact_blockers": ARTICLE_EXACT_BLOCKERS,
                "entrypoint": __file__,
                "experiment_name": experiment_name,
                "distributed_island_runners": _distributed_runners,
                **build_run_metadata(env_config, appo_run_config),
            },
            f,
            indent=4,
            default=str,
        )

    appo_config = build_article_appo_config(env_config, appo_run_config, mu_server=mu_server)
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
