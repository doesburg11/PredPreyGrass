import environments.predpreygrass_variable_energy_transfer as predpreygrass

from config.config_pettingzoo import (
    env_kwargs,
    training_steps_string,
    local_output_directory,
)

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import supersuit as ss
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils.conversions import parallel_wrapper_fn

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
from stable_baselines3.common.monitor import Monitor


def optimize_ppo(trial):
    # Define the environment
    parallel_env = parallel_wrapper_fn(env_fn.raw_env)

    # Train a single model to play as each agent in a parallel environment
    raw_parallel_env = parallel_env(render_mode=None, **env_kwargs)

    # create parallel environments by concatenating multiple copies of the base environment
    num_vec_envs_concatenated = 8
    raw_parallel_env = ss.pettingzoo_env_to_vec_env_v1(raw_parallel_env)
    raw_parallel_env = ss.concat_vec_envs_v1(
        raw_parallel_env,
        num_vec_envs_concatenated,
        num_cpus=8,
        base_class="stable_baselines3",
    )


    # Suggest hyperparameters
    n_steps = trial.suggest_int('n_steps', 2048, 4096)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1,log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.00000001, 0.1,log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)

    # Ensure batch_size is a factor of n_steps * num_vec_envs_concatenated
    total_buffer_size = n_steps * num_vec_envs_concatenated
    while total_buffer_size % batch_size != 0:
        n_steps += 1
        total_buffer_size = n_steps * num_vec_envs_concatenated

    # Create the PPO model
    model = PPO(
        'MlpPolicy', 
        raw_parallel_env, 
        verbose=0, 
        n_steps=n_steps, 
        gamma=gamma,
        batch_size=batch_size,
        learning_rate=learning_rate, 
        ent_coef=ent_coef, 
        clip_range=clip_range
        )

    # Define evaluation environment separately to avoid state leaks
    eval_env = parallel_wrapper_fn(env_fn.raw_env)(render_mode=None, **env_kwargs)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(
        eval_env,
        num_vec_envs_concatenated,
        num_cpus=8,
        base_class="stable_baselines3",
    )

    tune_results_dir = local_output_directory + "tune_results/logs/"

    # Set up evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=tune_results_dir,
                                 log_path=tune_results_dir, eval_freq=1000,
                                 deterministic=True, render=False)


    # Train the model
    try:
        model.learn(total_timesteps=1000000, callback=eval_callback)
    except AttributeError as e:
        print(f"Training failed: {e}")
        return None

    # Evaluate the trained model
    try:
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    except AttributeError as e:
        print(f"Evaluation failed: {e}")
        return None
    
    return mean_reward


# Create and run the study
if __name__ == "__main__":
    # Ensure the correct reference to the environment function
    env_fn = predpreygrass

    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_ppo, n_trials=100)

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))