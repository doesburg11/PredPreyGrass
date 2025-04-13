config_ppo = {
    'train_batch_size': 1024, 
    'gamma': 0.99, 
    'lr': 0.0003, 
    'num_gpus_per_learner': 1, 
    'num_learners': 1, 
    'num_env_runners': 1, 
    'num_envs_per_env_runner': 1, 
    'rollout_fragment_length': "auto", 
    'sample_timeout_s': 600, 
    'num_cpus_per_env_runner': 1, 
    'num_cpus_for_main_process': 1
}