from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
    local_output_root,
)
env_kwargs["is_parallel_wrapped"] = True # environment loop is parallel wrapped
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.trainer import (
    Trainer
)
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.linearization_weights_constructor import (
    construct_linearalized_weights
)

# external libraries
from momaland.utils.conversions import mo_aec_to_parallel
from momaland.utils.parallel_wrappers import LinearizeReward
from os.path import dirname as up
import os
import time
import shutil
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    n_steps = trial.suggest_int('n_steps', 128, 2048)

    # Define the number of predators and prey
    num_predators = env_kwargs["n_possible_predator"]
    num_prey = env_kwargs["n_possible_prey"]
    # Construct the weights
    weights = construct_linearalized_weights(num_predators, num_prey)

    # Initialize the environment
    env = mo_predpreygrass_v0.env(**env_kwargs)
    parallel_env = mo_aec_to_parallel(env)
    parallel_env = LinearizeReward(parallel_env, weights)

    # train the model
    trainer = Trainer(
        parallel_env,
        destination_output_dir,
        model_file_name,
        steps=training_steps,
        seed=0,
        **env_kwargs,
    )
    start_training_time = time.time()
    trainer.train()
    end_training_time = time.time()
    training_time = end_training_time - start_training_time

    trainer = Trainer(env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps)

    env = parallel_wrapper_fn(parallel_env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class='stable_baselines3')

    # Create the model
    model = PPO(
        MlpPolicy,
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        verbose=0
    )

    # Create the evaluation environment
    eval_env = mo_predpreygrass_v0.env(**env_kwargs)
    eval_env = mo_aec_to_parallel(eval_env)
    eval_env = LinearizeReward(eval_env, weights)
    eval_env = parallel_wrapper_fn(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(eval_env, 4, num_cpus=4, base_class='stable_baselines3')

    # Create the EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(total_timesteps=int(training_steps_string), callback=eval_callback)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

    return mean_reward

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))