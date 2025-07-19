Environment v2.4: with adjustment in config_train.py becaus of low reproduction of predators
Old: experiment_1
New: experiment_2

| Parameter                           | Old  | New  | Why                           |
| ----------------------------------- | ---- | ---- | ----------------------------- |
| `initial_energy_predator`           | 5.0  | 6.0  | Higher viability              |
| `energy_loss_per_step_predator`     | 0.08 | 0.06 | Slightly less harsh           |
| `max_energy_gain_per_prey`          | 3.5  | 5.0  | Hunting payoff more realistic |
| `reproduction_chance_predator`      | 0.85 | 0.95 | Less stochastic punishment    |
| `reproduction_energy_efficiency`    | 0.95 | 0.90 | Adds realistic energy cost    |
| `n_initial_active_speed_1_predator` | 10   | 12   | Higher initial diversity      |
| `n_initial_active_speed_2_predator` | 10   | 12   | Higher initial diversity      |
-------------------------------------------------------------------------------------
