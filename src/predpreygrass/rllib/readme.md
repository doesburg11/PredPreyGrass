# Decentralized training wit RLlib new API stack (version 2.40+ 2.44.1)

Step function in predpreygrass_rllib_env.py
────────────────────────────────────────────────────────────────────
Step 0: Setup
────────────────────────────────────────────────────────────────────
- Initialize rewards, terminateds, truncateds, infos dicts
- Reset episode truncation and termination flags

────────────────────────────────────────────────────────────────────
Step 1: Grass Energy Regeneration
────────────────────────────────────────────────────────────────────
- If grass patches exist:
    - Add energy gain to all patches (vectorized)
    - Clip energies to maximum
    - Update grid world at static grass positions

────────────────────────────────────────────────────────────────────
Step 2: Agent Movement, Energy Cost, Aging
────────────────────────────────────────────────────────────────────
- For each agent action:
    - If agent alive:
        - Move agent based on action
        - Subtract movement energy cost
        - Update agent position in grid world
        - Increment agent age
        - (Optional) Print movement log if enabled

────────────────────────────────────────────────────────────────────
Step 3: Check for Dead Agents
────────────────────────────────────────────────────────────────────
- For each agent:
    - If energy <= 0:
        - Mark as dead
        - Remove from grid world
        - Set termination flag
        - (Optional) Log death if enabled

────────────────────────────────────────────────────────────────────
Step 4: Handle Reproduction
────────────────────────────────────────────────────────────────────
- For each agent:
    - If energy > reproduction threshold:
        - Spawn new offspring agent
        - Random free spawn position
        - Set offspring energy, age, and speed
        - Update grid world
        - (Optional) Log reproduction if enabled

────────────────────────────────────────────────────────────────────
Step 5: Step Counter Increment & Truncation Check
────────────────────────────────────────────────────────────────────
- Increment step counter
- If step counter >= max_episode_steps:
    - Mark global truncation
- If all agents dead:
    - Mark global termination

────────────────────────────────────────────────────────────────────
Return:
────────────────────────────────────────────────────────────────────
- Observations
- Rewards
- Terminated flags
- Truncated flags
- Infos
