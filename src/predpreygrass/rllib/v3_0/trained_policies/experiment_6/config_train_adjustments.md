**Environment v2.4: **

experiment-1 - "exp-1" guess, see run_config.json

experiment-2 - with adjustment in config-train.py because of low reproduction of predators, see table-1

experiment-3 - n-possible-speed-2-agents = 0, because experiment-2 led to extiction of speed-2-agents. Simplify environment to concentrate on implementing first and second law of thermodynamics.

experiment-4 - raise n-possible-speed-1-predators from 30 to 50, raise n-possible_1_speed_1_prey from 40 to 60, because the n_active_predators hits the upperbound quickly and stays there (s)ee eval in exp-3 at checkpoint 1000) (n-initial_active_predators = 24)

experiment-5 - lowered n_possible_speed_1_predators from 50 to 35, lowered n-possible-speed-1-prey from 60 to 45, because the positive training results for predators is significantly less in experiment-4 compared to experiment-3. So trying to find out in experiment-5 if the predator training results improve versus experiment-4. By tightening the upperbound of n-active-speed-1-predators again by setting n-possible-speed-1-predators (35) closer to experiment-3 again (30).


experiment-6 - back to experiment-3, because experiment-5 also showed no significant improvement like experiment-3 did. Verifciatiom if that was (not) a lucky shot. It appears that indeed experiment-6 mimics experiment-3.So, keeping the n-active-predators restricted while less restriction on prey, means more food and thus rewards per predators.


table-1: differences between experiment-1 and experiment-2
--------------------------------------------------------------------------------------
| Parameter                           | exp-1 | exp-2 | Why                           |
| ----------------------------------- | ----- | ----- | ----------------------------- |
| `initial_energy_predator`           | 5.0   | 6.0   | Higher viability              |
| `energy_loss_per_step_predator`     | 0.08  | 0.06  | Slightly less harsh           |
| `max_energy_gain_per_prey`          | 3.5   | 5.0   | Hunting payoff more realistic |
| `reproduction_chance_predator`      | 0.85  | 0.95  | Less stochastic punishment    |
| `reproduction_energy_efficiency`    | 0.95  | 0.90  | Adds realistic energy cost    |
| `n_initial_active_speed_1_predator` | 10    | 12    | Higher initial diversity      |
| `n_initial_active_speed_2_predator` | 10    | 12    | Higher initial diversity      |
--------------------------------------------------------------------------------------
