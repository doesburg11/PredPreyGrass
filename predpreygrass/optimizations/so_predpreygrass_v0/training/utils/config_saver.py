import os


class ConfigSaver:
    def __init__(
        self,
        destination_training_file,
        environment_name,
        training_steps_string,
        env_kwargs,
        local_output_root,
        destination_source_code_dir,
    ):
        self.destination_training_file = destination_training_file
        self.environment_name = environment_name
        self.training_steps_string = training_steps_string
        self.env_kwargs = env_kwargs
        self.local_output_root = local_output_root
        self.destination_source_code_dir = destination_source_code_dir

    def save(self):
        # Save training scenario to file
        with open(self.destination_training_file, "w") as training_file:
            training_file.write("environment: " + self.environment_name + "\n")
            training_file.write("learning algorithm: PPO \n")
            training_file.write("training steps: " + self.training_steps_string + "\n")
            training_file.write("------------------------\n")
            for item in self.env_kwargs:
                training_file.write(
                    str(item) + " = " + str(self.env_kwargs[item]) + "\n"
                )
            training_file.write("------------------------\n")

        # Overwrite config file locally, with the parameters for the current scenario
        code = "local_output_root = '{}'\n".format(self.local_output_root)
        code += "training_steps_string = '{}'\n".format(self.training_steps_string)
        code += "env_kwargs = dict(\n"

        for key, value in self.env_kwargs.items():
            code += f"    {key}={value},\n"

        code += ")\n"

        config_env_file = "config_predpreygrass.py"
        config_file_directory = os.path.join(
            self.destination_source_code_dir, "envs/_so_predpreygrass_v0/config/"
        )

        with open(
            os.path.join(config_file_directory, config_env_file), "w"
        ) as config_file:
            config_file.write(code)
