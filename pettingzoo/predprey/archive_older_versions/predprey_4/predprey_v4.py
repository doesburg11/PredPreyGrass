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
FAILED SO FAR

"""

from pettingzoo.predprey.predprey_4.predprey import (
    env,
    parallel_env,
    raw_env,
)

__all__ = ["env", "parallel_env", "raw_env"]
