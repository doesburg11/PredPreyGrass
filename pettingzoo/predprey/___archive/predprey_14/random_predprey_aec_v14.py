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
9) [v9]
-renaming, simplify
-convert-ready to bi-agent (prey/grass or predator/prey)
-number the grid cell with (x,y) coordinates 
            GRID COORDINATES MOORE: x_size=y_size=16
            ------------------------------------------
            |(0,0)...........(12,0)...........(15,0) |
            |  .                .                .   |
            |  .                .                .   |
            |  .      (11,9) (12,9)  (13,9)      .   |
            |(0,10).. (11,10 (12,10) (13,10)..(15,10)|
            |  .      (11,11)(12,11) (13,11)     .   |
            |  .                .                .   |
            |  .                .                .   |
            |(0,15)..........(12,15)..........(15,15)|
            ------------------------------------------

                       action = [     0,       1,       2,       3,      4,      5,      6,     7,     8]
            self.motion_range = [[-1,-1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0],[-1,1],[0,1],[-1,-1]]
            ------------------------------------------
            |                 .                      |
            |                  .                     |
            |         (-1,-1) (0,-1) (1,-1)          |
            |         (-1,0   (0,0)  (1,0)           |
            |         (-1,1)  (0,1)  (1,1)           |
            |                  .                     |
            |                  .                     |
            |                                        |
            ------------------------------------------

            GRID COORDINATES VON NEUMANN: x_size=y_size=16
            ------------------------------------------
            |(0,0)...........(12,0)...........(15,0) |
            |  .                .                .   |
            |  .                .                .   |
            |  .             (12,9)              .   |
            |(0,10).. (11,10 (12,10) (13,10)..(15,10)|
            |  .             (12,11)             .   |
            |  .                .                .   |
            |  .                .                .   |
            |(0,15)..........(12,15)..........(15,15)|
            ------------------------------------------
                       action = [    0,       1,      2,     3,      4]
            self.motion_range = [[0,-1], [-1, 0], [1, 0], [0,1], [0, 0]]
            ------------------------------------------
            |                 .                      |
            |                  .                     |
            |                 (0,-1)                 |
            |         (-1,0   (0,0)  (1,0)           |
            |                 (0,1)                  |
            |                  .                     |
            |                  .                     |
            |                                        |
            ------------------------------------------

-change predators into prey_9 (pre with moore movements)

different actions range and observation range per agent?
    -at creation of a single agent: give observation as input to create attribute (self.observation_range)
    -varying action spaces Moore/not Moore does not work. Only option that does work is
    Moore=False for Both agents
10)[v10] 
-remove prey_9 again (former predators)
-draw_prey_instances (based on instance list rather than layer)
-draw_gras_instances (based on instance list rather than layer)
-draw agent_instance_id_nr (instead of using the array index nr)
10) [v11]
-store diverse observation spaces per agent into array
-this is accomplished by setting an overall max_observation_range(=7 for example)
 and setting an observation range per agent which is smaller or equal to this range.
 If range is maller (=5 for example) then the outer ring(s) of the max_observation_range
 are all set to zero. The observations all need identical shapes, so this shortcut is 
 used in this manner. 
 -removed shared_reward (had no function)
 -obtains reward from last() per agent
 -add to cumulative reward
 -use agent_selector.next() and agent_selector.is_last() to count-up n_cycles+=1
 12) [v12]
 -average n_cycle added
 -create_agents makes a standalone list. does not append to existing list anymore
 -self.prey1_instance list and self.prey2_instance_list added (prey=prey1+prey2). To be able to 
 create stats on both groups.
 13) [v13]
 -implement average reward stats  voor both types of prey
 -observation range made fully flexible below the max_observation_range
 -remove hard coded groups prey1_name_list and prey2_name_list and flexibilize
 -rename env into raw_env and pred_prey_env in sb3_predprey_vector.py
"""
"""
14) [v14]
-pixel-scale into kwargs
remove array-index related arrays and change them to dicts
 -self.rewards in PredPrey
 -self.grasses_gone
remove agent_idx number and replace by agent_id_name in predprey.step()

TODO 
and agent_layer. Change move_agent (line 125) from agent_idx to agent_instance
-pixel-scale dynamic on x_size/y?size?
-personalize reward per agent
-personalize (not)Moore per agent

"""

from predprey import raw_env
from pettingzoo.utils import agent_selector


env_kwargs = dict(
    render_mode="human", 
    max_cycles=10000, 
    x_size=16, 
    y_size=16, 
    n_prey1=4,
    n_prey2=4,
    n_grasses=30,
    max_observation_range=9,     
    obs_range_prey1=9,     
    obs_range_prey2=3,
    freeze_grasses=True, 
    catch_reward=5.0, 
    urgency_reward=-0.1,
    moore_neighborhood_prey=False,
    moore_neighborhood_grasses=False,
    pixel_scale=40
)

num_games = 10
if num_games >1:
    env_kwargs["render_mode"]="None"

env = raw_env(**env_kwargs) 

def average(rewards):
    N = len(rewards)
    avg_rewards = sum(rewards) / N
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
avg_cycles = [0 for _ in range(num_games)]
std_rewards = [0 for _ in range(num_games)]

agent_selector = agent_selector(agent_order=env.agents)

for i in range(num_games):
    env.reset(seed=i)
    agent_selector.reset()
    cumulative_rewards = {agent: 0.0 for agent in env.possible_agents}
    n_cycles = 0
    #print("env.agents ",env.agents)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        cumulative_rewards[agent] += reward
        if termination or truncation:
            action = None
        else:
            action = env.action_space(agent).sample()
        env.step(action)
        if agent_selector.is_last(): # called at end of cycle
            n_cycles += 1
            #print("cycle ",n_cycles," ",{key : round(cumulative_rewards[key], 1) for key in cumulative_rewards})
        agent_selector.next()   # called at end of cycle

    avg_rewards[i]= average(cumulative_rewards.values()) # type: ignore
    avg_cycles[i]= n_cycles
    std_rewards[i]= std_dev(cumulative_rewards, avg_rewards[i])
    print(f"Cycles = {n_cycles}", f"Avg = {round(avg_rewards[i],1)}", f"Std = {round(std_rewards[i],1)}",end=" ")
    print()
env.close()
print(f"Average of Avg = {round(average(avg_rewards),1)}")
print(f"Average of Cycles = {round(average(avg_cycles),1)}")
