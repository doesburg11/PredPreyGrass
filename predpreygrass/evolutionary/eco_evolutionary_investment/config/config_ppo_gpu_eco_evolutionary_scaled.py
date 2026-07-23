from predpreygrass.evolutionary.eco_evolutionary_investment.config.config_ppo_gpu_eco_evolutionary import config_ppo as _base_config_ppo

# Trial 6 (population scaling): the GPU inference/training batch for this new API
# stack batches all currently-live agents within an env step into one forward pass,
# so memory pressure scales roughly with total simultaneous agents = population-per-env
# x parallel-env-count. At the original 4x population target, num_envs_per_env_runner=3
# (21 envs) OOM'd, =2 (14 envs) reached 15.1/16.3GB (too tight to trust unattended).
#
# At the reduced 2x population target (see config_env_eco_evolutionary_scaled.py),
# =2 (14 envs) pilot ran stably at only 8.5/16.3GB (52%) through 20 iterations --
# comfortable headroom. num_envs_per_env_runner=3 (21 envs, the original un-scaled
# default) has a population x env product of 2x21=42, below the risky 4x/=2 point's
# 4x14=56, so it's expected to land safely under the ceiling (roughly 12-13GB
# estimated) while running close to the original R7 pace. See
# predpreygrass/evolutionary/RESULTS.md, Trial 6.
config_ppo = {
    **_base_config_ppo,
    "num_envs_per_env_runner": 3,
    "minibatch_size": 64,
}
