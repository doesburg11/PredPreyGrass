import numpy as np

from pettingzoo.predprey.predprey_2.utils.discrete_agent import DiscreteAgent

# create nagents agents

def create_agents(nagents, xs, ys, obs_range, randomizer, flatten=False, moore_neighborhood=True):
    """
    Initializes the agents on a map (map_matrix).
    -nagents: the number of agents to put on the map
    """
    agents = []
    for _ in range(nagents):
        xinit, yinit =  (randomizer.integers(0, xs), randomizer.integers(0, ys))      

        agent = DiscreteAgent(
            xs, ys, obs_range=obs_range, flatten=flatten, moore_neighborhood=moore_neighborhood
        )
        agent.set_position(xinit, yinit)
        agents.append(agent)
    return agents
