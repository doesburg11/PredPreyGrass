# reuse_actors=True
# Trial configuration table only with relevant hyperparameters

import os
os.environ["PYTHONWARNINGS"]="ignore::DeprecationWarning"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

import math
import argparse

from predpreygrass.rllib.hyper_parameter_tuning.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.hyper_parameter_tuning.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.hyper_parameter_tuning.utils.networks import build_multi_module_spec

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

from ray.rllib.algorithms.callbacks import DefaultCallbacks

def get_config_ppo():
    num_cpus = os.cpu_count()
    if num_cpus == 32:
        # Workaround to avoid PyTorch CUDA memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        from predpreygrass.rllib.hyper_parameter_tuning.config.config_ppo_gpu_pbt import config_ppo
    elif num_cpus == 8:
        from predpreygrass.rllib.hyper_parameter_tuning.config.config_ppo_cpu_pbt_smoke import config_ppo
    else:
        raise RuntimeError(f"Unsupported cpu_count={num_cpus}. Please add matching config_ppo.")
    return config_ppo

class ReproductionStatsCallback(DefaultCallbacks):
    """
    Collect per-episode reproduction counts for the target agent.
    We intercept episode end, read the agent's episode reward (raw), convert to
    reproduction count (reward / reproduction reward value), and store it in a list.
    The Trainable drains this list each step to update streaming stats.
    """
    TARGET_AGENT = "type_1_predator_0"

    def __init__(self):
        super().__init__()
        self._reproduction_counts = []  # holds floats (per-episode reproduction counts)

    def on_episode_end(self, *, episode, **kwargs):  # RLlib callback API
        # Depending on RLlib version, agent reward histories vary.
        # We use episode.agent_rewards (dict with ((agent_id, policy_id), reward_sum)).
        try:
            total_reward = 0.0
            found = False
            for (aid, _pid), rew in episode.agent_rewards.items():
                if aid == self.TARGET_AGENT:
                    total_reward = rew
                    found = True
                    break
            if not found:
                return
            # Convert raw reward to reproduction count. Use env config reproduction reward.
            reproduction_reward = config_env["reproduction_reward_predator"]["type_1_predator"]
            if reproduction_reward:
                reproduction_count = total_reward / reproduction_reward
                # Store raw reproduction count; Trainable will aggregate.
                self._reproduction_counts.append(reproduction_count)
        except Exception:
            pass  # fail-safe; do not break training


class RLLibPPOTrainable(Trainable):
    """Tune Trainable wrapping PPO with custom reproduction-based PBT metric."""

    TARGET_AGENT_FOR_PBT = "type_1_predator_0"

    def __init__(self, config=None, logger_creator=None, **kwargs):
        """Init defaults BEFORE parent constructor triggers setup().

        Ray's Trainable.__init__ invokes self.setup() which uses these fields.
        Extra kwargs (e.g. storage) are ignored for forward compatibility.
        """
        # Streaming reproduction stats (Welford accumulators)
        self._repr_n = 0
        self._repr_mean = 0.0
        self._repr_M2 = 0.0
        # Risk adjustment defaults
        self._risk_adjustment_enabled = True
        self._risk_adjustment_factor = 0.5
        self._risk_switch_n = 30
        # Fallback / diagnostics
        self._debug_repro_callback_hits = 0
        self._grace_iterations_before_fallback = 2
        self._use_episode_return_fallback = True

        super().__init__(config, logger_creator)

    # Welford online update
    def _welford_update(self, x: float):
        self._repr_n += 1
        delta = x - self._repr_mean
        self._repr_mean += delta / self._repr_n
        delta2 = x - self._repr_mean
        self._repr_M2 += delta * delta2

    def _welford_stats(self):
        if self._repr_n < 2:
            return self._repr_mean, 0.0, None
        var = self._repr_M2 / (self._repr_n - 1)
        return self._repr_mean, math.sqrt(var), var

    def setup(self, config):
        # Keep the full tune config for future resets
        self._cfg = dict(config)
        # Pull optional risk adjustment flags from config
        self._risk_adjustment_enabled = self._cfg.get("risk_adjustment_enabled", True)
        self._risk_adjustment_factor = self._cfg.get("risk_adjustment_factor", 0.5)
        self._risk_switch_n = self._cfg.get("risk_adjustment_switch_n", 30)
        # Fallback/grace overrides
        self._grace_iterations_before_fallback = self._cfg.get("grace_iterations_before_fallback", self._grace_iterations_before_fallback)
        self._use_episode_return_fallback = self._cfg.get("use_episode_return_fallback", self._use_episode_return_fallback)

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
                #lr=cfg["lr"],
                # clip_param=cfg["clip_param"],
                # entropy_coeff=cfg["entropy_coeff"],
                num_epochs=cfg["num_epochs"],
                minibatch_size=cfg["minibatch_size"],
                train_batch_size_per_learner=cfg["train_batch_size_per_learner"],
            )
        )

        # Now build
        if getattr(base_conf, "callbacks_class", None) is None:
            base_conf.callbacks(ReproductionStatsCallback)
        self.algo = base_conf.build_algo()

    def step(self):
        result = self.algo.train()
        used_target_agent_reward = False

        # --- Robust metric extraction for PBT ---
        candidate_key_paths = [
            ["pbt_metric"],
            ["episode_return_mean"],
            ["episode_reward_mean"],
            ["env_runners/episode_return_mean"],
            ["env_runners/episode_reward_mean"],
            ["env_runners", "episode_return_mean"],
            ["env_runners", "episode_reward_mean"],
            ["evaluation", "episode_return_mean"],
            ["evaluation", "episode_reward_mean"],
            ["sampler_results", "episode_reward_mean"],
        ]

        def _get_by_path(d, path):
            cur = d
            if len(path) == 1 and "/" in path[0]:
                parts = path[0].split("/")
            else:
                parts = path
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return None
            return cur

        metric_val = None
        for path in candidate_key_paths:
            metric_val = _get_by_path(result, path)
            if metric_val is not None:
                break

        if metric_val is None:
            env_runners_section = result.get("env_runners")
            if isinstance(env_runners_section, dict):
                rollout_sec = env_runners_section.get("rollout")
                if isinstance(rollout_sec, dict):
                    metric_val = rollout_sec.get("episode_return_mean") or rollout_sec.get("episode_reward_mean")

                if metric_val is None:
                    agent_ret = env_runners_section.get("agent_episode_returns_mean")
                    if isinstance(agent_ret, dict) and agent_ret:
                        target_key = self.TARGET_AGENT_FOR_PBT
                        if target_key in agent_ret:
                            metric_val = agent_ret[target_key]
                            used_target_agent_reward = True
                        else:
                            metric_val = sum(agent_ret.values()) / len(agent_ret)

        if used_target_agent_reward and metric_val is not None:
            try:
                reproduction_reward = config_env["reproduction_reward_predator"]["type_1_predator"]
            except Exception:
                reproduction_reward = 10.0
            if reproduction_reward:
                metric_val = metric_val / reproduction_reward
                result.setdefault("custom_metrics", {})
                result["custom_metrics"]["target_agent_episode_reward_mean"] = metric_val * reproduction_reward
                result["custom_metrics"]["target_agent_avg_reproductions"] = metric_val
        if metric_val is None or (isinstance(metric_val, float) and (metric_val != metric_val)):
            if not hasattr(self, "_warned_missing_metric"):
                print("[RLLibPPOTrainable] Could not locate episode return metric. Top-level keys:", list(result.keys()))
                if "env_runners" in result:
                    print("[RLLibPPOTrainable] env_runners keys:", list(result["env_runners"].keys()) if isinstance(result["env_runners"], dict) else type(result["env_runners"]))
                self._warned_missing_metric = True
            metric_val = 0.0
        # Fallback recorded only (not used for pbt_metric until reproductions exist)
        result.setdefault("custom_metrics", {})
        result["custom_metrics"].setdefault("fallback_episode_return_mean_raw", float(metric_val))

        # --- Per-episode reproduction statistics (streaming) ---
        callback_obj = getattr(self.algo, "callbacks", None)
        new_counts = []
        if callback_obj and hasattr(callback_obj, "_reproduction_counts"):
            if callback_obj._reproduction_counts:
                self._debug_repro_callback_hits += 1
            new_counts = callback_obj._reproduction_counts
            callback_obj._reproduction_counts = []
        for rc in new_counts:
            self._welford_update(rc)

        mean_r, std_r, var_r = self._welford_stats()

        cm = result["custom_metrics"]  # guaranteed above
        reproduction_metric = None
        selected_source = 0  # 2=reproduction,1=agent_reward,0=episode_return

        # 1. Reproduction-based metric (preferred when available)
        if self._repr_n > 0:
            risk_active = False
            if (self._risk_adjustment_enabled and self._repr_n >= self._risk_switch_n and std_r > 0):
                reproduction_metric = mean_r - self._risk_adjustment_factor * std_r
                risk_active = True
            else:
                reproduction_metric = mean_r
            cm["target_agent_reproductions_mean"] = mean_r
            cm["target_agent_reproductions_mean_raw"] = mean_r
            cm["target_agent_reproductions_risk_adjusted"] = reproduction_metric
            cm["target_agent_risk_adjusted_reproductions"] = reproduction_metric
            cm["target_agent_risk_adjustment_active"] = 1.0 if risk_active else 0.0
            cm["target_agent_reproductions_std"] = std_r
            cm["target_agent_reproductions_n"] = self._repr_n
            if self._repr_n > 1 and std_r > 0:
                ci_half = 1.96 * std_r / math.sqrt(self._repr_n)
                cm["target_agent_reproductions_ci_low"] = mean_r - ci_half
                cm["target_agent_reproductions_ci_high"] = mean_r + ci_half
            else:
                cm["target_agent_reproductions_ci_low"] = mean_r
                cm["target_agent_reproductions_ci_high"] = mean_r
            selected_source = 2
        else:
            # If not available yet, still expose empty stats for consistency
            cm.setdefault("target_agent_reproductions_n", self._repr_n)

        # 2. Agent reward normalized fallback (independent of reproduction availability)
        agent_returns = None
        env_runners = result.get("env_runners")
        if isinstance(env_runners, dict):
            agent_returns = env_runners.get("agent_episode_returns_mean")
        normalized_agent_reward = None
        if isinstance(agent_returns, dict):
            raw_agent_mean = agent_returns.get(self.TARGET_AGENT_FOR_PBT)
            if raw_agent_mean is not None:
                try:
                    reproduction_reward = config_env["reproduction_reward_predator"]["type_1_predator"]
                except Exception:
                    reproduction_reward = 10.0
                if reproduction_reward:
                    normalized_agent_reward = raw_agent_mean / reproduction_reward
                else:
                    normalized_agent_reward = float(raw_agent_mean)
                cm["target_agent_episode_reward_mean_raw"] = float(raw_agent_mean)
                cm["target_agent_reward_normalized"] = float(normalized_agent_reward)

        # 3. Episode return fallback already logged as fallback_episode_return_mean_raw
        episode_return_fallback = cm.get("fallback_episode_return_mean_raw")

        # Select pbt_metric with grace period: reproduction > normalized_agent_reward > episode_return
        training_iter = result.get("training_iteration", 0)
        grace_active = training_iter < self._grace_iterations_before_fallback
        if reproduction_metric is not None:
            result["pbt_metric"] = float(reproduction_metric)
        else:
            if grace_active:
                # keep neutral until after grace (unless reproduction already present)
                result["pbt_metric"] = 0.0
                cm["pbt_metric_grace_active"] = 1.0
            else:
                if normalized_agent_reward is not None:
                    result["pbt_metric"] = float(normalized_agent_reward)
                    selected_source = 1
                elif self._use_episode_return_fallback and episode_return_fallback is not None:
                    result["pbt_metric"] = float(episode_return_fallback)
                    selected_source = 0
                else:
                    result["pbt_metric"] = 0.0

        # Record all source variants explicitly for analysis
        if reproduction_metric is not None:
            cm["pbt_metric_reproduction"] = float(reproduction_metric)
        if normalized_agent_reward is not None:
            cm["pbt_metric_agent_reward_normalized"] = float(normalized_agent_reward)
        if episode_return_fallback is not None:
            cm["pbt_metric_episode_return"] = float(episode_return_fallback)
        cm["pbt_metric_selected_source"] = float(selected_source)
        # Diagnostics
        cm["reproduction_callback_hits"] = float(self._debug_repro_callback_hits)
        cm["reproduction_counts_observed"] = float(self._repr_n)

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
    # -----------------------------
    # CLI & ENV OVERRIDES
    # -----------------------------
    parser = argparse.ArgumentParser(description="Run PPO PBT for PredPreyGrass with optional risk metric tuning.")
    parser.add_argument("--risk-enabled", type=str, choices=["true", "false"], help="Force risk adjustment enabled/disabled (overrides tuning).")
    parser.add_argument("--risk-factor", type=float, help="Force a specific risk adjustment factor (overrides tuning).")
    parser.add_argument("--risk-switch-n", type=int, help="Force reproduction count threshold to activate risk adjustment.")
    parser.add_argument("--tune-risk-flags", action="store_true", help="Allow tuning of risk parameters (unless individually overridden).")
    parser.add_argument("--grace-iters", type=int, help="Iterations to wait before enabling non-reproduction fallbacks (default 2).")
    parser.add_argument("--disable-episode-return-fallback", action="store_true", help="Disable the final episode-return fallback metric.")
    cli_args, unknown = parser.parse_known_args()

    # Environment variable overrides (lower precedence than explicit CLI)
    env_enabled = os.getenv("RISK_ADJUSTMENT_ENABLED")
    env_factor = os.getenv("RISK_ADJUSTMENT_FACTOR")
    env_switch = os.getenv("RISK_ADJUSTMENT_SWITCH_N")
    env_grace = os.getenv("GRACE_ITERS_BEFORE_FALLBACK")
    env_disable_episode_return = os.getenv("DISABLE_EPISODE_RETURN_FALLBACK")

    def _parse_bool(v):
        if v is None:
            return None
        return str(v).lower() in ("1", "true", "yes", "y")

    override_enabled = None
    if cli_args.risk_enabled is not None:
        override_enabled = _parse_bool(cli_args.risk_enabled)
    elif env_enabled is not None:
        override_enabled = _parse_bool(env_enabled)

    override_factor = None
    if cli_args.risk_factor is not None:
        override_factor = float(cli_args.risk_factor)
    elif env_factor is not None:
        try:
            override_factor = float(env_factor)
        except ValueError:
            pass

    override_switch_n = None
    if cli_args.risk_switch_n is not None:
        override_switch_n = int(cli_args.risk_switch_n)
    elif env_switch is not None:
        try:
            override_switch_n = int(env_switch)
        except ValueError:
            pass

    tune_risk = cli_args.tune_risk_flags
    # Grace iterations override resolution
    override_grace_iters = None
    if cli_args.grace_iters is not None:
        override_grace_iters = int(cli_args.grace_iters)
    elif env_grace is not None:
        try:
            override_grace_iters = int(env_grace)
        except ValueError:
            pass

    # Episode return fallback enable/disable
    disable_episode_return_fallback = False
    if cli_args.disable_episode_return_fallback:
        disable_episode_return_fallback = True
    elif env_disable_episode_return is not None:
        disable_episode_return_fallback = env_disable_episode_return.lower() in ("1","true","yes","y")

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
    # Prepare metadata for this run (also captures whether certain params will be tuned)
    effective_grace_default = 2
    grace_meta = (
        override_grace_iters if override_grace_iters is not None else [0, 1, 2, 3, 5]
    )  # list ==> indicates tuning else int
    fallback_meta = (
        False if disable_episode_return_fallback else [True, False]
    )  # list indicates tuning, bool indicates fixed

    risk_meta = {
        "enabled": override_enabled if override_enabled is not None else ("TUNED" if cli_args.tune_risk_flags else True),
        "factor": override_factor if override_factor is not None else ("TUNED" if cli_args.tune_risk_flags else 0.5),
        "switch_n": override_switch_n if override_switch_n is not None else ("TUNED" if cli_args.tune_risk_flags else 30),
    }

    config_metadata = {
        "config_env": config_env,
        "config_ppo": config_ppo,
        "metric_pipeline": {
            "priority_order": ["reproduction", "normalized_agent_reward", "episode_return"],
            "grace_iterations_before_fallback": grace_meta,
            "episode_return_fallback": fallback_meta,
            "risk_adjustment": risk_meta,
        },
    }
    with open(experiment_path / "run_config.json", "w") as f:
        json.dump(config_metadata, f, indent=4)

    # Create a sample environment to build (multi) module specs
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
    # Base set of hyperparameters we always allow to mutate.
    hyperparam_mutations = {
        "lr": lambda: random.choice(config_ppo["pbt_lr_choices"]),
        # Additional candidates (clip, entropy) can be re-enabled once reward scale is stable.
        "num_epochs": lambda: random.randint(*config_ppo["pbt_num_epochs_range"]),
        "minibatch_size": lambda: random.choice(config_ppo["pbt_minibatch_choices"]),
        "train_batch_size_per_learner": lambda: random.choice(config_ppo["pbt_train_batch_size_choices"]),
    }
    # Expose fallback related knobs to PBT only if user did NOT hard override them.
    if override_grace_iters is None:
        grace_choices = [0, 1, 2, 3, 5]
        hyperparam_mutations["grace_iterations_before_fallback"] = lambda: random.choice(grace_choices)
    if not disable_episode_return_fallback:
        # allow toggling on/off; if user disabled explicitly we keep it off
        hyperparam_mutations["use_episode_return_fallback"] = lambda: random.choice([True, False])

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
    # Build param_space with conditional risk tuning
    param_space = {
        "algo_config": ppo_config,                # base template
        "lr": tune.choice(config_ppo["pbt_lr_choices"]),
        "num_epochs": tune.choice(config_ppo["pbt_num_epochs_range"]),
        "minibatch_size": tune.choice(config_ppo["pbt_minibatch_choices"]),
        "train_batch_size_per_learner": tune.choice(config_ppo["pbt_train_batch_size_choices"]),
    }

    # Grace iterations before fallback: fixed if overridden else a tune.choice to enable trial diversity.
    if override_grace_iters is not None:
        param_space["grace_iterations_before_fallback"] = override_grace_iters
    else:
        param_space["grace_iterations_before_fallback"] = tune.choice([0, 1, 2, 3, 5])

    # Episode return fallback toggle: respect explicit disable else allow tuning.
    if disable_episode_return_fallback:
        param_space["use_episode_return_fallback"] = False
    else:
        param_space["use_episode_return_fallback"] = tune.choice([True, False])

    # Risk parameter choices (used only if tuning and not overridden)
    RISK_ENABLED_CHOICES = [True, False]
    RISK_FACTOR_CHOICES = [0.25, 0.5, 0.75, 1.0]
    RISK_SWITCH_CHOICES = [10, 20, 30, 40, 50, 75, 100]

    # Enabled flag
    if override_enabled is not None:
        param_space["risk_adjustment_enabled"] = override_enabled
    elif tune_risk:
        param_space["risk_adjustment_enabled"] = tune.choice(RISK_ENABLED_CHOICES)
    else:
        param_space["risk_adjustment_enabled"] = True

    # Factor
    if override_factor is not None:
        param_space["risk_adjustment_factor"] = override_factor
    elif tune_risk:
        param_space["risk_adjustment_factor"] = tune.choice(RISK_FACTOR_CHOICES)
    else:
        param_space["risk_adjustment_factor"] = 0.5

    # Switch N
    if override_switch_n is not None:
        param_space["risk_adjustment_switch_n"] = override_switch_n
    elif tune_risk:
        param_space["risk_adjustment_switch_n"] = tune.choice(RISK_SWITCH_CHOICES)
    else:
        param_space["risk_adjustment_switch_n"] = 30

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

    # --- Extended metric pipeline summary (custom metrics) ---
    # We stored multiple candidate metrics; here we show which source actually fed pbt_metric.
    custom = best_result.metrics.get("custom_metrics", {}) or {}
    source_code = custom.get("pbt_metric_selected_source")
    source_map = {2: "reproduction", 1: "normalized_agent_reward", 0: "episode_return"}
    source_label = source_map.get(int(source_code) if source_code is not None else -1, "unknown")

    summary_block = {
        "pbt_metric_final": best_result.metrics.get("pbt_metric"),
        "pbt_metric_source_code": source_code,
        "pbt_metric_source_label": source_label,
        # Reproduction stats (if available)
        "reproductions_mean": custom.get("target_agent_reproductions_mean"),
        "reproductions_std": custom.get("target_agent_reproductions_std"),
        "reproductions_n": custom.get("target_agent_reproductions_n"),
        "risk_adjustment_active": custom.get("target_agent_risk_adjustment_active"),
        "risk_adjusted_reproductions": custom.get("target_agent_risk_adjusted_reproductions"),
        # Fallback components (help diagnose if reproduction never appeared)
        "agent_reward_normalized": custom.get("pbt_metric_agent_reward_normalized"),
        "episode_return_fallback": custom.get("pbt_metric_episode_return"),
        # Diagnostic counters
        "reproduction_callback_hits": custom.get("reproduction_callback_hits"),
    }
    print("\nMetric pipeline summary (final iteration of best trial):\n")
    pprint.pprint(summary_block)
    print("\nInterpretation: 'pbt_metric_source_label' shows which signal drove selection."
          " If it is not 'reproduction', reproduction data was unavailable or still in grace period.")
