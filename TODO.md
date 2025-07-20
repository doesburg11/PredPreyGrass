**To do**

- Move experiments out of repo with exception of the first

- can v2_5 be generalized to v1_0?
    - vx_x reduces to a config
    - experiment: vx_x + config_env

- Fitness parameters:
    - offspring per agent
    - offspring per agent per energy
- Protocol for storing/retrieving stats per step (outside env: in evaluatio loop)
- Flowcharts Visio-like
    - In Linux?
    - Windows emulator?

- Find a mechanism to limit energy intake for

    - The grass at full 2.0 can regarded as a battery; extra solar power does not increase the capcity of the battery any more.

    - Maybe distinction between solarpower on the one hand which can be transformed to grass (eg. When all grass is zero and the regeneration rate is 0.08. if 100 grass available that soloarpower transforms into 0,08x100 = 8 units per time step (at maximum). On the other hand the real regeneration energy turned into grass energy.

    - The latter needs to be recorded still.

Record for agents to be used in tooltips:

    - Unique ID (other than eg. predator_0, which can be reused)

    - Age
