# changes compared to dev_1_works.py
# reuse_actors=True
# Trial configuration table only with relevant hyperparameters

import os
os.environ["PYTHONWARNINGS"]="ignore::DeprecationWarning"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

from predpreygrass.rllib.v3_1.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.v3_1.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.v3_1.utils.networks import build_multi_module_spec

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.tune import Tuner
from ray.tune import Trainable, RunConfig, CheckpointConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import PlacementGroupFactory

import pprint
import json
import random
from datetime import datetime
from pathlib import Path

class RLLibPPOTrainable(Trainable):
    """
    Tune Trainable that builds an RLlib PPO Algorithm and supports reuse_actors via reset_config.
    Expect a dict config with:
      - "algo_config": an RLlib PPOConfig (already chained with .environment/.training/...)
      - optional "restore_from_path": checkpoint path to restore weights (string)
    """

    def setup(self, config):
        # Keep the full tune config for future resets
        self._cfg = dict(config)

        # Build or rebuild the RLlib Algorithm
        self._build_algo(self._cfg)

        # If we’re starting with a checkpoint, restore once.
        ckpt = self._cfg.pop("restore_from_path", None)
        if ckpt:
            self.algo.restore_from_path(ckpt)

    def _build_algo(self, cfg):
        base_conf: PPOConfig = cfg["algo_config"].copy(copy_frozen=False)  # ← unfrozen copy

        # Pull top-level sampled values from the Tune config and apply to the config copy
        # (These keys exist because we put them into param_space.)
        base_conf = (
            base_conf
            .training(
                lr=cfg["lr"],
                clip_param=cfg["clip_param"],
                entropy_coeff=cfg["entropy_coeff"],
                num_epochs=cfg["num_epochs"],
                minibatch_size=cfg["minibatch_size"],
                train_batch_size_per_learner=cfg["train_batch_size_per_learner"],
            )
        )

        # Now build
        self.algo = base_conf.build_algo()

    def step(self):
        result = self.algo.train()

        # Normalize RLlib’s moving target metric names into one stable key
        metric_aliases = [
            "episode_return_mean",                 # often present (newer stacks)
            "env_runners/episode_return_mean",     # older/new-API internal
            "evaluation/episode_return_mean",      # if you evaluate
        ]
        for k in metric_aliases:
            if k in result:
                result["pbt_metric"] = result[k]
                break
        else:
            # Ensure the key exists so PBT(require_attrs=True) never crashes
            result["pbt_metric"] = float("nan")

        return result


    def save_checkpoint(self, checkpoint_dir):
        # RLlib will create a subdir; return the path for Tune
        return self.algo.save_to_path(checkpoint_dir)

    def load_checkpoint(self, path):
        # Restore full RLlib Algorithm state
        self.algo.restore_from_path(path)

    def reset_config(self, new_config):
        """
        Called by Tune when PBT exploits/explores with reuse_actors=True.
        We rebuild the Algorithm from a fresh copy of the base PPOConfig,
        applying any mutated top-level values (lr, clip, entropy, epochs, minibatch, train_bs).
        """
        try:
            # Keep the Tune config (contains top-level sampled/mutated values)
            self._cfg = dict(new_config)

            # If a checkpoint to clone from is provided, capture and remove it from cfg
            ckpt = None
            for k in ("restore_from_path", "checkpoint_path", "checkpoint_dir"):
                if self._cfg.get(k):
                    ckpt = self._cfg.pop(k)
                    break

            # IMPORTANT: stop the old algo before rebuilding
            try:
                if hasattr(self, "algo") and self.algo is not None:
                    self.algo.stop()
            except Exception:
                pass

            # Rebuild with the *new* values now in self._cfg
            self._build_algo(self._cfg)

            # If exploitation asked us to load parent weights, restore now
            if ckpt:
                self.algo.restore_from_path(ckpt)
                print(f"[reset_config] restored from {ckpt}")

            print(f"[reset_config] succeeded (actor reused). pid={os.getpid()}")
            return True
        except Exception as e:
            print(f"[reset_config] failed: {e}")
            return False

    def cleanup(self):
        try:
            self.algo.stop()
        except Exception:
            pass


def env_creator(config):
    return PredPreyGrass(config)


def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        # Workaround to avoid PyTorch CUDA memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        from predpreygrass.rllib.v3_1.config.config_ppo_gpu_pbt import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.v3_1.config.config_ppo_cpu_pbt_smoke import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo


def policy_mapping_fn(agent_id, *args, **kwargs):
    parts = agent_id.split("_")
    type = parts[1]
    role = parts[2]
    return f"type_{type}_{role}"


# Postprocess the perturbed PBT config to ensure it's still valid
def explore(config):
    # cap/repair top-level keys
    config["train_batch_size_per_learner"] = min(config["train_batch_size_per_learner"], 2048)
    if config["train_batch_size_per_learner"] < config["minibatch_size"] * 2:
        config["train_batch_size_per_learner"] = config["minibatch_size"] * 2

    config["num_epochs"] = min(max(int(config["num_epochs"]), 1), 30)
    return config


if __name__ == "__main__":
    register_env("PredPreyGrass", env_creator)

    # experiment output directory
    ray_results_dir = "~/Dropbox/02_marl_results/predpreygrass_results/ray_results/"
    ray_results_path = Path(ray_results_dir).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"PPO_PBT_{timestamp}"
    experiment_path = ray_results_path / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Get the PPO config based on the number of CPUs and save into experiment
    config_ppo = get_config_ppo()
    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)

    sample_env = env_creator(config=config_env)


    # Group spaces per policy id (first agent of each policy defines the space)
    obs_by_policy, act_by_policy = {}, {}
    for agent_id, obs_space in sample_env.observation_spaces.items():
        pid = policy_mapping_fn(agent_id)
        if pid not in obs_by_policy:
            obs_by_policy[pid] = obs_space
            act_by_policy[pid] = sample_env.action_spaces[agent_id]

    # Build one MultiRLModuleSpec in one go
    multi_module_spec = build_multi_module_spec(obs_by_policy, act_by_policy)

    # Policies dict for RLlib
    policies = {
        pid: (None, obs_by_policy[pid], act_by_policy[pid], {})
        for pid in obs_by_policy
    }

    ppo_config = (
        PPOConfig()
        .environment(env="PredPreyGrass", env_config=config_env)
        .framework("torch")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            # These params are tuned from a fixed starting value.
            clip_param=config_ppo["clip_param"],
            lr=config_ppo["lr"],
            entropy_coeff=config_ppo["entropy_coeff"],
            # set safe defaults; will be overridden per-trial
            num_epochs=config_ppo["num_epochs"],
            minibatch_size=config_ppo["minibatch_size"],
            train_batch_size_per_learner=config_ppo["train_batch_size_per_learner"],
        )
        .rl_module(rl_module_spec=multi_module_spec)
        .learners(
            num_gpus_per_learner=config_ppo["num_gpus_per_learner"],
            num_learners=config_ppo["num_learners"],
            num_cpus_per_learner=0,   # ← add this
        )
        .env_runners(
            num_env_runners=config_ppo["num_env_runners"],
            num_envs_per_env_runner=config_ppo["num_envs_per_env_runner"],
            rollout_fragment_length=config_ppo["rollout_fragment_length"],
            sample_timeout_s=config_ppo["sample_timeout_s"],
            num_cpus_per_env_runner=config_ppo["num_cpus_per_env_runner"],
        )
        .resources(
            num_cpus_for_main_process=config_ppo["num_cpus_for_main_process"],
        )
    )

    # PBT setup
    hyperparam_mutations = {
        "lr": lambda: random.choice(config_ppo["pbt_lr_choices"]),
        "clip_param": lambda: random.uniform(*config_ppo["pbt_clip_range"]),
        "entropy_coeff": lambda: random.choice(config_ppo["pbt_entropy_choices"]),
        "num_epochs": lambda: random.randint(*config_ppo["pbt_num_epochs_range"]),
        "minibatch_size": lambda: random.choice(config_ppo["pbt_minibatch_choices"]),
        "train_batch_size_per_learner": lambda: random.choice(config_ppo["pbt_train_batch_size_choices"]),
    }

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=config_ppo["perturbation_interval"], 
        resample_probability=config_ppo["resample_probability"],
        quantile_fraction=config_ppo["quantile_fraction"],
        hyperparam_mutations=hyperparam_mutations,  # Specifies the mutations of hyperparams
        custom_explore_fn=explore,
        log_config=False,
        metric="pbt_metric",
        mode="max",
        require_attrs=True,  

    )
    # Pack everything the Trainable needs into param_space
    param_space = {
        "algo_config": ppo_config,                # base template
        # ← tunables live at top-level so Tune can sample them
        # "lr": tune.choice(config_ppo["pbt_lr_choices"]),
        # "clip_param": tune.uniform(*config_ppo["pbt_clip_range"]),
        # "entropy_coeff": tune.choice(config_ppo["pbt_entropy_choices"]),
        "num_epochs": tune.randint(*config_ppo["pbt_num_epochs_range"]),
        "minibatch_size": tune.choice(config_ppo["pbt_minibatch_choices"]),
        "train_batch_size_per_learner": tune.choice(config_ppo["pbt_train_batch_size_choices"]),
    }

    # Stopping criteria
    stopping_criteria = {"training_iteration": config_ppo["max_iters"]}

    checkpoint_every = 1

    # --- Placement group (PG) resources for one trial ---
    # Bundle 0: trial driver + learners
    main_cpus = (
        config_ppo["num_cpus_for_main_process"]
        + config_ppo["num_learners"] * 0   # keep 0 if you don't give CPUs to learners
    )

    # one bundle per runner
    bundles = [{"CPU": main_cpus}]
    for _ in range(config_ppo["num_env_runners"]):
        bundles.append({"CPU": config_ppo["num_cpus_per_env_runner"]})

    pgf = PlacementGroupFactory(bundles=bundles, strategy="PACK")

    tuner = Tuner(
        tune.with_resources(RLLibPPOTrainable, resources=pgf),  # ← two bundles
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=pbt,
            num_samples=config_ppo["pbt_num_samples"],
            reuse_actors=True,
        ),
        run_config=RunConfig(
            name=experiment_name,
            storage_path=str(ray_results_path),
            stop=stopping_criteria,
            callbacks=None,
            checkpoint_config=CheckpointConfig(
                num_to_keep=100,
                checkpoint_frequency=checkpoint_every,
                checkpoint_at_end=True,
            ),
      ),
    )

    result = tuner.fit()

    best_result = result.get_best_result(
        metric="env_runners/episode_return_mean",
        mode="max",
    )
    print(best_result.metrics.get("env_runners/episode_return_mean"))

    print("Best performing trial's final set of hyperparameters:\n")
    pprint.pprint({k: v for k, v in best_result.config.items() if k in hyperparam_mutations})

    env_metrics = best_result.metrics.get("env_runners", {})

    metrics_to_print = {
        "episode_return_mean": env_metrics.get("episode_return_mean"),
        "episode_return_max": env_metrics.get("episode_return_max"),
        "episode_return_min": env_metrics.get("episode_return_min"),
        "episode_len_mean": env_metrics.get("episode_len_mean"),
    }
    print("\nBest performing trial's final reported metrics:\n")
    pprint.pprint(metrics_to_print)
