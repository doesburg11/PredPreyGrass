# discretionary libraries
from predpreygrass.envs import so_predpreygrass_v0
from predpreygrass.envs._so_predpreygrass_v0.config.so_config_predpreygrass import (
    env_kwargs,
    training_steps_string,
    local_output_root,
)

# external libraries
import os
import sys
import time
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
import optuna


def optimize_ppo(trial):
    # Define the environment
    env = so_predpreygrass_v0.env(render_mode=None, **env_kwargs)
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # create parallel environments by concatenating multiple copies of the base environment
    num_vec_envs_concatenated = 8
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs_concatenated,
        num_cpus=8,
        base_class="stable_baselines3",
    )
    # env = Monitor(env)  # Wrap the evaluation environment with Monitor

    # default values
    verbose = 0
    # learning_rate = 3e-4
    # n_steps =2048
    # batch_size = 64
    # gamma = 0.99
    gae_lambda = 0.95
    # clip_range = 0.2
    # ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    n_epochs = 10
    normalize_advantage = True

    # Suggest hyperparameters
    n_steps = trial.suggest_int("n_steps", 2048, 4096)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.8, 0.9999)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.15, 0.2)

    # Ensure batch_size is a factor of n_steps * num_vec_envs_concatenated
    total_buffer_size = n_steps * num_vec_envs_concatenated
    while total_buffer_size % batch_size != 0:
        n_steps += 1
        total_buffer_size = n_steps * num_vec_envs_concatenated

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        n_steps=n_steps,
        gamma=gamma,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        normalize_advantage=normalize_advantage,
    )

    # Create the evaluation environment
    eval_env = so_predpreygrass_v0.env(**env_kwargs)
    eval_env = aec_to_parallel(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    eval_env = ss.concat_vec_envs_v1(
        eval_env, num_vec_envs_concatenated, num_cpus=4, base_class="stable_baselines3"
    )
    # eval_env = Monitor(eval_env)  # Wrap the evaluation environment with Monitor

    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=tune_results_logs_dir,
        log_path=tune_results_logs_dir,
        eval_freq=1000,
        deterministic=True,
        render=False,
    )

    # Train the model
    try:
        model.learn(total_timesteps=training_steps, callback=eval_callback)
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
    N_TRIALS = 1
    environment_name = "so_predpreygrass_v0"
    # Ensure the correct reference to the environment function
    training_steps = int(training_steps_string)

    start_time = str(time.strftime("%Y-%m-%d_%H:%M:%S"))  # add seconds
    file_name = f"{environment_name}"
    tune_results_dir = local_output_root + "tune_results/" + start_time
    tune_results_logs_dir = tune_results_dir + "/logs/"

    # save the source code locally
    python_file_name = os.path.basename(sys.argv[0])
    python_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    file_names_in_directory = os.listdir(python_directory)
    # create the destination directory for the source code
    os.makedirs(tune_results_dir, exist_ok=True)

    # Copy all files and directories in the current directory to the local directory
    # for safekeeping experiment scenarios
    for item_name in file_names_in_directory:
        source_item = os.path.join(python_directory, item_name)
        destination_item = os.path.join(tune_results_dir, item_name)

        if os.path.isfile(source_item):
            shutil.copy2(source_item, destination_item)
        elif os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)

    start_training_time = time.time()
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=N_TRIALS)
    end_training_time = time.time()
    print(f"Training time: {end_training_time - start_training_time}")

    # Open a file to save the screen output
    with open(tune_results_logs_dir + "/screenoutput.txt", "w") as f:
        # Redirect standard output to the file
        sys.stdout = f

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        df = study.trials_dataframe()
        df.to_csv(tune_results_logs_dir + "/optuna_trials.csv", index=False)

    # Reset standard output to its original value
    sys.stdout = sys.__stdout__
    print("Optimization finished")

    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)
