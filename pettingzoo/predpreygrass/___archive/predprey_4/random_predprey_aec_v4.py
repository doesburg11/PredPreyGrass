from pettingzoo.predpreygrass.predprey_4 import predprey_v4

env_kwargs = dict(
    render_mode="human", 
    max_cycles=500, 
    x_size=16, 
    y_size=16, 
    shared_reward=False, 
    n_evaders=30,
    n_pursuers=8,
    obs_range=7, 
    n_catch=1,
    freeze_evaders=True, 
    tag_reward=0.01,
    catch_reward=5.0, 
    urgency_reward=-0.1, 
    surround=False, 
    moore_neighborhood=True
)

#print(env_kwargs)
num_games = 10
if num_games >1:
    env_kwargs["render_mode"]="None"

env = predprey_v4.env(**env_kwargs)

env.reset(seed=42)

#print(env.agents)

def average_dict(rewards):
    N = len(rewards.values())
    avg_rewards = sum(rewards.values()) / N
    return avg_rewards

def average_array(avgrewards):
    N = len(avgrewards)
    avg_rewards = sum(avgrewards) / N
    return avg_rewards

def std_dev(rewards, avg_reward):
    N = len(rewards.values())
    variance_rewards = 0
    for _agent in rewards:
        variance_rewards += pow(rewards[_agent]-avg_reward,2)
    variance_rewards = 1/(N-1)*variance_rewards
    std_rewards = pow(variance_rewards,0.5)
    return std_rewards

avg_rewards = [0 for _ in range(num_games)]
std_rewards = [0 for _ in range(num_games)]

for i in range(num_games):
    env.reset(seed=i)
    rewards = {agent: 0.0 for agent in env.possible_agents}
    n_cycles = 0
    for agent in env.agent_iter():
        #print("agent ", agent)
        if agent=="pursuer_0":
            n_cycles += 1
        observation, reward, termination, truncation, info = env.last()
        for a in env.agents:
            rewards[a] += env.rewards[a]

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()
            #print("agent ", agent," takes action ",action)

        env.step(action)
    avg_rewards[i]= average_dict(rewards)
    std_rewards[i]= std_dev(rewards, avg_rewards[i])
    print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
env.close()
#print(avg_rewards)
print(f"Average of Avg = {round(average_array(avg_rewards),1)}")
