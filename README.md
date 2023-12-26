

 ### Design restrictions
 1. All Observation ranges must be equal (PPO)
 2. All Observation spaces must be equal
 3. All action spaces must be equal

 #### workarounds:
 ad 1. Implement an overall maximum observation range and a specific (smaller) observation range per agent by zero-ing all non-observable cells.
 ad 2. Implement an overall maximum observation space. In this case a specific observation channel can have at the mo
 ad 3. Implement an overall max_n_possible_actions.


