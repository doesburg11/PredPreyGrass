**Environment v2.4: **

experiment_1 - "exp1" guess, see run_config.json

experiment_2 - with adjustment in config_train.py because of low reproduction of predators, see table_1

experiment_3 - n_possible_speed_2_agents = 0, because experiment_2 led to extiction of speed_2_agents.
               Simplify environment to concentrate on implementing first and second law of thermodynamics.

experiment_4 - raise n_possible_speed_1_predators from 30 to 50, raise n_possible_1_speed_1_prey from 40 to 60,
               because the n_active_predators hits the upperbound quickly and stays there (s)ee eval in exp_3 at checkpoint 1000)
               (n_initial_active_predators = 24)

experiment_5 - lowered n_possible_speed_1_predators from 50 to 35, lowered n_possible_1_speed_1_prey from 60 to 45,
               because the positive training results for predators is significantly less in experiment_4 compared to experiment_3.
               So trying to find out in experiment_5 if the predator training results improve versus experiment_4. By tightening the
               upperbound of n_active_speed_1_predators again by setting n_possible_speed_1_predators (35) closer to experiment_3 again (30).


experiment_6 - back to experiment_3, because experiment_5 also showed no significant improvement like experiment_3 did. verifcatiom if that
               was a lucky shot.


table_1: differences between experiment_1 and experiment_2
--------------------------------------------------------------------------------------
| Parameter                           | exp_1 | exp_2 | Why                           |
| ----------------------------------- | ----- | ----- | ----------------------------- |
| `initial_energy_predator`           | 5.0   | 6.0   | Higher viability              |
| `energy_loss_per_step_predator`     | 0.08  | 0.06  | Slightly less harsh           |
| `max_energy_gain_per_prey`          | 3.5   | 5.0   | Hunting payoff more realistic |
| `reproduction_chance_predator`      | 0.85  | 0.95  | Less stochastic punishment    |
| `reproduction_energy_efficiency`    | 0.95  | 0.90  | Adds realistic energy cost    |
| `n_initial_active_speed_1_predator` | 10    | 12    | Higher initial diversity      |
| `n_initial_active_speed_2_predator` | 10    | 12    | Higher initial diversity      |
--------------------------------------------------------------------------------------
