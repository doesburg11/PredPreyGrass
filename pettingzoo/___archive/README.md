### Versions and changes descriptions saved in Archive

[v0] 

a Moore neighborhood with additional parameter moore_neighborhood = False
as default

[v1]

parameres xb, yb for changing the size of the center non inhabitable white square

[v2] 

simplify the code

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

[v4] integrate files
-integrate discrete_agent.py into predprey_base.py

-integrate predprey_base.py into predprey.py

-more specific les abstract coding

-remove Agent from DiscreteAgent inheritance

-create n_action_pursuers from the moore or von neumann choice, so that reset does not have to be called 
beforehand

-Both evaders and pursuers can move in Moore or Von Neumann directions

[v6]

-surround option removed

-n_catch removed

[v7]

-created new agent predators and renamed evaders to grass and pursuers to prey

-added predators to grid but not yet activated

[v8]

-remove predprey_v8.py

-remove wrappers

-add parallel_env to executables (random and training)

-remove local_ratio(=1)

-remove unused methods in class PredPrey and DiscreteAgent

-simplify safely_observe

[v9]

-renaming, simplify

-convert-ready to bi-agent (prey/grass or predator/prey)

-number the grid cell with (x,y) coordinates 
            GRID COORDINATES MOORE: x_grid_size=y_grid_size=16
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
            self.motion_range = [[-1,-1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0],[-1,1],[0,1],[1,1]]
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

            GRID COORDINATES VON NEUMANN: x_grid_size=y_grid_size=16
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


[v10] 

-remove prey_9 again (former predators)

-draw_prey_instances (based on instance list rather than layer)

-draw_gras_instances (based on instance list rather than layer)

-draw agent_instance_id_nr (instead of using the array index nr)

[v11]

-store diverse observation spaces per agent into array

-this is accomplished by setting an overall max_observation_range(=7 for example)
 and setting an observation range per agent which is smaller or equal to this range.
 If range is maller (=5 for example) then the outer ring(s) of the max_observation_range
 are all set to zero. The observations all need identical shapes, so this shortcut is 
 used in this manner. 

 -removed shared_reward (had no function)

 -obtains reward from last() per agent

 -add to cumulative reward

 -use agent_selector.next() and agent_selector.is_last() to count-up n_aec_cycles+=1

[v12]

 -average n_cycle added

 -create_agents makes a standalone list. does not append to existing list anymore

 -self.prey1_instance list and self.prey2_instance_list added (prey=prey1+prey2). To be able to 
 create stats on both groups.

[v13]

 -implement average reward stats  voor both types of prey

 -observation range made fully flexible below the max_observation_range

 -remove hard coded groups prey1_name_list and prey2_name_list and flexibilize

 -rename env into raw_env and pred_prey_env in sb3_predprey_vector.py

[v14]

-pixel-scale into kwargs
remove array-index related arrays and change them to dicts

 -self.rewards in PredPrey

 -self.grass_gone

[v15]

-remove agent_idx number and replace by agent_name in predprey.step()
and agent_layer. Change move_agent (line 125) from agent_idx to agent_instance.

-cleanup

[v16]

-urgency_reward and catch_reward  'personalized' and moved to DisrceteAgent and renamed to homeostatic_energy_per_aec_cycle
and catch_grass_reward respectively

-removed agent_name_to_id_nr_dict (can be fetched by agent_name_to_instance_dict.agent_id_nr)

-removed urgency_reward and catch_reward from pred_prey args

[v17]

-let pred_prey handle the agent_intialization of self.agents (=pred_prey_env.agent_names_list)

-rename self.frames to self.n_aec_cycles and use in main program (remove agent_selector.last())

-surpress warnings by quick fixes vscode when possible

[v18]

-remove RandomPolicy and SingleActionPolicy controller classes

[v19]

-clean up DiscreteAgent

[v20]

-return agent_selector to show end of cycle section in main program, can help to show a rewards after cycle, which
went wrong earlier

-rewards as args in DiscreteAgent

-rename Prey1 to Predator

-rename Prey2 to Prey

-cleanup,make better distinction between predprey and raw_env corresponding variables

-hard coded number of channels to 3 in order to add "predator" in 'agent_type_names'

-removed self.agents in PredPrey (apparently no use)

[v21]

-changed revisions text to 'predprey.py'

-Color change in graphs

[v22]

-switch colors observations pred and prey

-set id_nr x position at 3.4 for all agents

-add observation

-rename agent_name_list to agent_name_list

-implement: prey_layer and predator_layer

-predators do not remove anything

[v23]

-refinemnents of [v22]

[v24]

-Major breakthrough: Predator can eat Prey and Prey is subsequently removed from
'prey_instance_list'. This list can be used to watch the prey_names still allive.
NB 1: about removal agents:
AssertionError: expected agent prey_7 got termination or truncation agent predator_0. 
Parallel environment wrapper expects all agent death (setting an agent's self.terminations or 
self.truncations entry to True) to happen only at the end of a cycle.
NB 2: about removal agents
Knights-Archer-Zombies: This environment allows agents to spawn and die, 
so it requires using SuperSuit's Black Death wrapper, which provides blank 
observations to dead agents rather than removing them from the environment.

[v25]

-fixed termination at no_grass_left, on top of no_prey_left

-self.terminations are not to be used used by removal prey_agents: gives to much 
trouble in other lists/dicts 

-method that converts 'prey_instance_list' to 'pred_name_list' into 'pred_pred_env'. 
Because 'prey_name_list' itself cannot be reduced (due to PPO), 
but the 'prey_instance_list on the other hand can be reduced.

[v26]
-observations of prey_not_alive are fully set to zero for each element in compliance with black death of SuperSuit

-implemented rewards for predators to catch prey. this results in the emergent behavior
of prey fleeing predators 


[v27]

-implement simultaneous Moore/Von Neumann neighborhood for preditor/prey, 
(seperately is only possible with action masking; noy yet implemented)

-generalize action space to larger area and keep Moore/Von Neumann option
-progress bar implemented for PPO training

-link to TensorBoard to gauge performance 
https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode


To start a TensorBoard session from VSC:


Open the command palette (Ctrl/Cmd + Shift + P)


Search for the command “Python: Launch TensorBoard” and press enter.

You will be able to select the folder where your TensorBoard log files are located. 

By default, the current working directory will be used.

VSCode will then open a new tab with TensorBoard and its lifecycle will be managed 
by VS Code as well. This means that to kill the TensorBoard process all you 
have to do is close the TensorBoard tab.

-change x_size to x_grid_size and change y_size to y_grid_size

-change prey/predator_name_list to possible_predator/prey_name_list 
to emphasize this list does not change in run time

[v29]

-save every step to file in order to produce gif

[v30]

-delete method save_image

-implement self.save_image_steps = False

-fixed observation range error

-implement array (shape: 1xn_observation_channels) of dicts with keys as coordinates 

and values an array of agent_names. Gives per coordinate teh specific agent_names in it.

For more efficient searching.

[v31]

-make a dict (instead of action_range_iterator) whichs maps actions to movement fails. likely
because of discrepancies between numpy arrays and dicts, so motion_range remains in
its existing from

[v32]

-implement energy loss per predator
-implement starvation predator

[v34]

-implement bar chart which tracks energy levels per agent/type of agent

-death of Predator (and Prey) by starvation (implement minimum energy levels

-tried A2C with 4 predators and 8 prey, but it did not learn well

[v35]

-implement self.grass_position_dict and self.prey_instance_in_grid_location_dict for more 

efficient removal of grass and prey

-stop when all prey are eaten OR all predators are dead [HAS HUGE 

IMPACT ON LEARNING]

[v36]

-integrate draw functions in render function

-integrate position dicts 

[v37]

-remove as much as possible from AgentLayer class

-change n_predator etc to n_initial_predator etc, to use in reset

[v38]

-removed agent_layer class

[v39]

-remove flatten = false in observation space
