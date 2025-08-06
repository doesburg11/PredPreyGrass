import random

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.schedulers import PopulationBasedTraining
import pprint

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # Ensure enough data to do at least 2 minibatches per epoch
        if config.get("train_batch_size_per_learner", 0) < config.get("minibatch_size", 1) * 2:
            config["train_batch_size_per_learner"] = config["minibatch_size"] * 2

        # Ensure at least 1 epoch
        if config.get("num_epochs", 0) < 1:
            config["num_epochs"] = 1

        return config

    hyperparam_mutations = {
        "clip_param": lambda: random.uniform(0.01, 0.5),
        "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        "num_epochs": lambda: random.randint(1, 30),
        "minibatch_size": lambda: random.randint(128, 16384),
        "train_batch_size_per_learner": lambda: random.randint(2000, 160000),
    }

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
    )

    # Stop when we've either reached 100 training iterations or reward=300
    stopping_criteria = {"training_iteration": 3, "episode_reward_mean": 300}

    config = (
        PPOConfig()
        .environment("Humanoid-v5")
        .env_runners(num_env_runners=4)
        .training(
            # These params are tuned from a fixed starting value.
            kl_coeff=1.0,
            lambda_=0.95,
            clip_param=0.2,
            lr=1e-4,
            # These params start off randomly drawn from a set.
            num_epochs=tune.choice([10, 20, 30]),
            minibatch_size=tune.choice([128, 512, 2048]),
            train_batch_size_per_learner=tune.choice([10000, 20000, 40000]),
        )
        .rl_module(
            model_config=DefaultModelConfig(free_log_std=True),
        )
    )

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="env_runners/episode_return_mean",
            mode="max",
            scheduler=pbt,
            num_samples=1 if args.smoke_test else 2,
        ),
        param_space=config,
        run_config=tune.RunConfig(stop=stopping_criteria),
    )
    results = tuner.fit()

    best_result = results.get_best_result()

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
