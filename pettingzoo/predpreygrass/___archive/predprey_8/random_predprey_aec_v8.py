"""
PD: implements, compared to pursuit_v4: 
1) [v0] a Moore neighborhood with additional parameter moore_neighborhood = False
as default
2) [v1] parameres xb, yb for changing the size of the center non inhabitable white square
3) [v2] simplify the code
-removed manual policy
-removed optional rectangular obstacle in center and all possible obstacle variations in two_d_maps.py
-removed two_d_maps.py
-removed map_matrix
-model_state remained untouched (base) because it is input for observation; 
 model_state[0] has become obsolete, model_state[3] was already obsolete?
-removed agent_utils.py and integrated create_agents() into predprey_base.py)
-integrated Agent into discrete_agent.py and removed _utils_2.py
-integrated AgentLayer into discrete_agent.py and removed agent_layer.py
-integrated controllers into discrete_agent.py and removed controllers.py
-moved discrete_agent.py one directory level lower and removed directory utils
4) [v4] integrate files
-integrate discrete_agent.py into predprey_base.py
-integrate predprey_base.py into predprey.py
5) [v4] more specific les abstract coding
-remove Agent from DiscreteAgent inheritance
-create n_action_pursuers from the moore or von neumann choice, so that reset does not have to be called 
beforehand
-Both evaders and pursuers can move in Moore or Von Neumann directions
6) [v6]
-surround option removed
-n_catch removed
7) [v7]
-created new agent predators and renamed evaders to grass and pursuers to prey
-added predators to grid but not yet activated
8) [v8]
-remove predprey_v8.py
-remove wrappers
-add parallel_env to executables (random and training)
-remove local_ratio(=1)
-remove unused methods in class PredPrey and DiscreteAgent
-simplify safely_observe
"""

from pettingzoo.predpreygrass.predprey_8.predprey import raw_env

env_kwargs = dict(
    render_mode="human", 
    max_cycles=10000, 
    x_size=16, 
    y_size=16, 
    shared_reward=False, 
    n_predators=5,
    n_prey=8,
    n_grasses=30,
    obs_range=7, 
    freeze_grasses=True, 
    catch_reward=5.0, 
    urgency_reward=-0.1,
    moore_neighborhood_predators=False,
    moore_neighborhood_prey=False,
    moore_neighborhood_grasses=False
)


num_games = 10
if num_games >1:
    env_kwargs["render_mode"]="None"

env = raw_env(**env_kwargs)  

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

from pettingzoo.utils import agent_selector
agent_selector = agent_selector(agent_order=env.agents)


for i in range(num_games):
    env.reset(seed=i)
    agent_selector.reset()
    rewards = {agent: 0.0 for agent in env.possible_agents}
    n_cycles = 0
    for agent in env.agent_iter():
        #print("agent ",agent," is last is ",agent_selector.is_last())
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

        if agent_selector.is_last(): # called at end of cycle
            n_cycles += 1
        agent_selector.next()   # called at end of cycle

    avg_rewards[i]= average_dict(rewards)
    std_rewards[i]= std_dev(rewards, avg_rewards[i])
    print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
env.close()
#print(avg_rewards)
print(f"Average of Avg = {round(average_array(avg_rewards),1)}")
