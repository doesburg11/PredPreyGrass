"""
📊 Summary of Prey Training Results with tensorboard: "config_ppo_gpu_pbt_1.py"

From your prey graph:

Several trials clearly surpass a return of 300 and even approach 400 — notably the green and purple lines.

Lower-performing trials (e.g., orange) stabilize around 150–200.

Trials with high learning rates (e.g., lr=0.01) still manage to perform well — suggesting robustness.

Trial 00005 (clip_param=0.225, lr=0.0001, entropy=0.0, batch_size=2048, epochs=30) performs best — hinting that a longer training cycle with a larger batch size and moderate clip is beneficial.

🧠 Cross-Comparison: Predator vs. Prey
Observation	Predator	Prey
Highest returns	~100	~400
Best clip param range	0.22 – 0.29	0.22 – 0.30
Best learning rate	0.0001 – 0.001	0.0001
Entropy coeff	Low entropy works fine	Low entropy also ok
Train batch size	1024 – 2048	2048 best
Num epochs	10 – 30 (higher is better)	30 preferred
Minibatch	128–256	256 preferred
✅ Recommended PBT Ranges (Next Round)
hyperparam_mutations = {
    "lr": lambda: random.choice([1e-4, 5e-4, 1e-3]),
    "clip_param": lambda: random.uniform(0.2, 0.3),  # narrowed range
    "entropy_coeff": lambda: random.choice([0.0, 0.0001]),
    "train_batch_size_per_learner": lambda: random.choice([1024, 2048]),
    "minibatch_size": lambda: random.choice([128, 256]),
    "num_epochs": lambda: random.choice([10, 30]),  # keep high values
}


This narrows down the search to promising regions without losing diversity.

🛠️ Additional Suggestions

Increase perturbation interval to at least 6:

Let trials evolve more before being judged.

Fix metric access as discussed:

Make sure episode_reward_mean is surfaced at the top level in results.

Log trial configs to trace what mutations occurred (you already seem to have log_config=True, good).

✍️ Final Note

Your current setup is quite successful — both species show variation, emergence of dominant trials, and stable dynamics over 500 iterations. You’re ready for a focused refinement round, especially with tuned ranges and ensured exploitability.

Let me know if you'd like a Markdown block summarizing these parameter ranges for your documentation.
"""


config_ppo = {
    "max_iters": 2,
    # Core learning
    "lr": 5e-5,
    "gamma": 0.99,
    "lambda_": 0.95,
    "train_batch_size_per_learner": 2048,
    "minibatch_size": 128,
    "num_epochs": 6,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources
    "num_learners": 1,
    "num_env_runners": 4,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 0,
    "num_cpus_for_main_process": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
    # KL / exploration
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # pbt parameters
    "pbt_num_samples": 4,
    "perturbation_interval": 3,
    "resample_probability": 0.25,
    "quantile_fraction": 0.25,
    # PBT mutation ranges/choices
    "pbt_lr_choices": [1e-4, 5e-4, 1e-3],
    "pbt_clip_range": [0.2, 0.3],              
    "pbt_entropy_choices": [0.0, 0.0001],
    "pbt_num_epochs_range": [10, 30],          
    "pbt_minibatch_choices": [128, 256],
    "pbt_train_batch_size_choices": [1024, 2048],

}
