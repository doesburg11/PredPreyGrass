import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from predpreygrass.optimizations.mo_predpreygrass_v0.training.utils.logger import SampleLoggerCallback

from pettingzoo.utils.conversions import aec_to_parallel


class Trainer:
    def __init__(
            self, env_fn, 
            output_directory: str, 
            model_file_name: str,
            steps: int = 10_000, 
            seed: int = 0, 
            **env_kwargs):
        self.env_fn = env_fn
        self.output_directory = output_directory
        self.model_file_name = model_file_name
        self.steps = steps
        self.seed = seed
        self.env_kwargs = env_kwargs

    def train(self):

        #parallel_env = parallel_wrapper_fn(self.env_fn.raw_env)
        parallel_env = self.env_fn

        # Train a single model to play as each agent in a parallel environment
        #raw_parallel_env = parallel_env(render_mode=None, **self.env_kwargs)
        parallel_env.reset(seed=self.seed)

        print(f"Starting training on {str(parallel_env.metadata['name'])}.")
        # create parallel environments by concatenating multiple copies of the base environment
        num_vec_envs_concatenated = 8
        parallel_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
        parallel_env = ss.concat_vec_envs_v1(
            parallel_env,
            num_vec_envs_concatenated,
            num_cpus=8,
            base_class="stable_baselines3",
        )


        model = PPO(
            MlpPolicy,
            parallel_env,
            verbose=0,  # 0 for no output, 1 for info messages, 2 for debug messages, 3 default
            batch_size=256,
            tensorboard_log=self.output_directory + "/ppo_predprey_tensorboard/",
        )

        sample_logger_callback = SampleLoggerCallback()

        model.learn(
            total_timesteps=self.steps, progress_bar=True, callback=sample_logger_callback
        )
        saved_directory_and_model_file_name = self.output_directory + self.model_file_name + ".zip" 
        model.save(saved_directory_and_model_file_name)   

        print("saved model to: ", saved_directory_and_model_file_name)
        print("Model has been saved.")
        print(f"Finished training on {str(parallel_env.unwrapped.metadata['name'])}.")

        parallel_env.close()

