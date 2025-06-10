# discretionar libraries
from predpreygrass.pettingzoo.train.utils.logger import SampleLoggerCallback

# external libraries
import os
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils.conversions import aec_to_parallel


class Trainer:
    def __init__(
        self,
        env_fn,
        output_directory: str,
        model_file_name: str,
        steps: int = 10_000,
        seed: int = 0,
        hyperparameters: dict = None,
        **env_kwargs,
    ):
        self.env_fn = env_fn
        self.output_directory = output_directory
        self.model_file_name = model_file_name
        self.steps = steps
        self.seed = seed
        self.env_kwargs = env_kwargs

        # Set default hyperparameters and update with user-provided values
        self.hyperparameters = {
            "n_steps": 2048,
            "batch_size": 32_768,
            "n_epochs": 10,
            "verbose": 0,  # Default verbosity
        }
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

    def _prepare_environment(self, is_wrapped: bool, num_cores: int):
        # Choose environment type
        if is_wrapped:
            env = self.env_fn.raw_env(render_mode=None, **self.env_kwargs)
            # Convert AECEnv to ParallelEnv
            env = aec_to_parallel(env)
        else:
            env = self.env_fn.parallel_env(render_mode=None, **self.env_kwargs)

        env.reset(seed=self.seed)
        num_vec_envs_concatenated = num_cores

        # Concatenate vectorized environments
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(
            env,
            num_vec_envs_concatenated,
            num_cpus=num_cores,
            base_class="stable_baselines3",
        )
        return env

    def _train_model(self, raw_parallel_env, total_timesteps):
        print(f"Starting training on {str(raw_parallel_env.unwrapped.metadata['name'])}.")
        model = PPO(
            MlpPolicy,
            raw_parallel_env,
            n_steps=self.hyperparameters["n_steps"],
            batch_size=self.hyperparameters["batch_size"],
            n_epochs=self.hyperparameters["n_epochs"],
            verbose=self.hyperparameters["verbose"],
            tensorboard_log=self.output_directory + "/ppo_predpreygrass_tensorboard/",
        )
        sample_logger_callback = SampleLoggerCallback()
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
            callback=sample_logger_callback,
        )

        save_path = f"{self.output_directory}/{self.model_file_name}.zip"
        model.save(save_path)

        print(f"Saved model to: {save_path}")
        print("Model has been saved.")
        print(f"Finished training on {str(raw_parallel_env.unwrapped.metadata['name'])}.")

        raw_parallel_env.close()

    def train(self, is_wrapped: bool = True):
        num_cores = os.cpu_count()
        # Adjust training parameters based on available cores
        if num_cores == 128:  # Google Cloud Platform
            total_timesteps = 13_107_200
            self.hyperparameters["n_steps"] = 1_024
            self.hyperparameters["batch_size"] = 209_408
        elif num_cores == 8:  # Local Machine
            total_timesteps = self.steps
            self.hyperparameters["n_steps"] = 2_048
            self.hyperparameters["batch_size"] = 32_768
        elif num_cores == 2:  # Google Colab
            total_timesteps = 4_096_000
            self.hyperparameters["n_steps"] = 2_048
            self.hyperparameters["batch_size"] = 8_192
        else:  # Default
            total_timesteps = self.steps

        print(f"Number of CPU cores utilized: {num_cores}")

        # Prepare environment
        env = self._prepare_environment(is_wrapped=is_wrapped, num_cores=num_cores)

        # Train model
        self._train_model(env, total_timesteps)
