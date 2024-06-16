### Configurations

The benchmark configuration `config_pettingzoo_benchmark_1.py` is, somewhat arbitrarily, to test new developments in the environment. It is the same configuration used in an early stage of the environment some time ago and is displayed on the front page of the repository. It is used to test if it is still performing as expected when devoloping new features, such as for example creating new learning agents (Predator and Prey) or regrowing Grass agents.

<p align="center"><i>The benchmark configuration config_pettingzoo_benchmark_1.py</i></p>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/gif/predpreygrass_benchmark_0.gif" width="700" height="300"/>
</p>

opThe benchmark configuration `config_pettingzoo_benchmark_2.py` is, a more optimizes configuration combined with the possible creation and removal of Predators and Prey during runtime (upt to a certain maximum give by `n_possible_predator` and `n_possible_prey`). Grass agents are removed when eaten by Prey but regrow after a certain amount of cycles (depending on `energy_gain_per_step_grass` and `initial_energy_grass`).

#### Emergent behavior 

Overall, the configurations display emergent behavior of the agents:

***Predators***: 

- Predators are pursuing Prey, when in their observation range. When no Prey is in the Predator's observation range a common strategy for Predators is to hover around grass agents to wait for incoming Prey. An example of this behavior is shown in the video above by Predator number 0, who is hovering around at grass number 26, 31 and 41 before it is going after Prey number 12, when it enters the observation range of Predator 0. However, this strategy seems less frequent when `energy_gain_per_step_predator` becomes more negative, incentivizing Predators to abandon the 'wait-and-see' approach. 

- Predators pursue Prey even when there is no reward for eating Prey in the reward function and/or when there is no negative step reward. In such cases, they apparantly learn to consume Prey as a means to reproduce in the future, a process that requires sufficient energy. In that case, Predators learn to consume prey without the promise of immediate rewards, but attain only a (sparse) reward for reproduction.

- Predators sometimes engage into cooperation with another Predator to catch a single Prey. However, the configurations only gives a reward to the one Predator who ultimately catches the Prey. Therefore, this cooperating behavior can only be attributed to the higher overall probability per agent to catch a Prey in tandem.

***Prey***:

- Prey try to escape attacking Predators. Moreover, Prey apparently find out the observation range of Predators and sometimes try to stay out outside the Predator's observation range. This is obviously only the case if Prey have a larger observation range than Predators.

- Since the prey and predators strictly move within a Von Neumann neighborhood (left/right/up/down/stay) rather than a Moore neighborhood, it might be tempting to assume that prey can consistently outmaneuver a given predator, ensuring it is never caught. However this is under the conditions that: 
1) the agents move in a turn-based fashion
2) no other predators are involved in catching that specific prey; ie. no cooperation among Predators. 

This theoretical escape possibility is because a Prey, even when at risk of being caught in the immediate next turn by a predator, can always make a single step towards a position where the threatening predator needs to move at least two times to catch it (a feature of the Von Neumann neighborhood which is utilized).

However, this particular Prey behavior goes largely unnoticed in practice in the benchmark display because the simulation is trained in a *parallel* environment where all agents decide simultaneously, rather than in a turn-based fashion (note: the simulation is parallel trained but evaluated in a turn based fashion). 

At first glance, Prey let themselves sometimes easily get caught by Predators. This is maybe due to that no penalty is given for "dying". When a (severe) penalty for being eaten by Predators is given, then Prey tend to be more careful and stay out of the observation range of Predators more often.

This is illustrated by tuning the `death_reward_prey` parameter in the configuration:  

<p align="center"><i>Evading behavior Prey towards Predators can be enforced by penalizing death for Prey</i></p>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/readme/death_reward_prey_v_average_age_agents.png" width="450" height="270"/>
</p>

In any case, Prey are trying to escape from Predators, even when the penalty for dying is set to zero. This is because the "ultimate" reward is for reproduction for Prey, which is optimized when Prey are evading Predators. However, if an additional penalty for dying is introduced, Prey will try to avoid Predators even more. This can be concluded from a rising average age of Prey when the penalty is increased to -6. However, thereafter the system breaks. This is additionally illustrated below by a sudden collapse of the average episode length: 

<p align="center"><i>Gradual parameter variation can lead to radical shifts in outcomes</i></p>
<p align="center">
    <img src="https://github.com/doesburg11/PredPreyGrass/blob/main/assets/images/readme/death_reward_prey_v_episode_length.png" width="450" height="270"/>
</p>


