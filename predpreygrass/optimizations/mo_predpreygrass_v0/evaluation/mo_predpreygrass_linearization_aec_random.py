from predpreygrass.envs import mo_predpreygrass_v0
from predpreygrass.envs._mo_predpreygrass_v0.config.mo_config_predpreygrass import env_kwargs

from momaland.utils.aec_wrappers import LinearizeReward



env = mo_predpreygrass_v0.env(render_mode='human', **env_kwargs)

# TODO: remove hard coding of weights
weights={
    "predator_0": [0.5, 0.5],
    "predator_1": [0.5, 0.5],
    "predator_2": [0.5, 0.5],
    "predator_3": [0.5, 0.5],
    "predator_4": [0.5, 0.5],
    "predator_5": [0.5, 0.5],
    "predator_6": [0.5, 0.5],
    "predator_7": [0.5, 0.5],
    "predator_8": [0.5, 0.5],
    "predator_9": [0.5, 0.5],
    "predator_10": [0.5, 0.5],
    "predator_11": [0.5, 0.5],
    "predator_12": [0.5, 0.5],
    "predator_13": [0.5, 0.5],
    "predator_14": [0.5, 0.5],
    "predator_15": [0.5, 0.5],
    "predator_16": [0.5, 0.5],
    "predator_17": [0.5, 0.5],
    "prey_18": [0.5, 0.5],
    "prey_19": [0.5, 0.5],
    "prey_20": [0.5, 0.5],
    "prey_21": [0.5, 0.5],
    "prey_22": [0.5, 0.5],
    "prey_23": [0.5, 0.5],
    "prey_24": [0.5, 0.5],
    "prey_25": [0.5, 0.5],
    "prey_26": [0.5, 0.5],
    "prey_27": [0.5, 0.5],
    "prey_28": [0.5, 0.5],
    "prey_29": [0.5, 0.5],
    "prey_30": [0.5, 0.5],
    "prey_31": [0.5, 0.5],
    "prey_32": [0.5, 0.5],
    "prey_33": [0.5, 0.5],
    "prey_34": [0.5, 0.5],
    "prey_35": [0.5, 0.5],
    "prey_36": [0.5, 0.5],
    "prey_37": [0.5, 0.5],
    "prey_38": [0.5, 0.5],
    "prey_39": [0.5, 0.5],
    "prey_40": [0.5, 0.5],
    "prey_41": [0.5, 0.5],
}

env = LinearizeReward(env, weights)

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if reward > 0.0:
        print(f"agent: {agent}, reward: {reward}")
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
env.close()