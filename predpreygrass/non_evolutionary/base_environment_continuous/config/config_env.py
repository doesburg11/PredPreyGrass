"""Environment settings. Edit this dictionary before running train.py.

The values are those of the milestone world (runs AW-17 to AW-22 in
add_walls); environment.DEFAULT_CONFIG holds the same defaults. See SPEC.md
for what each setting does.
"""

config_env = {
    "width": 800,
    "height": 800,
    # Physics steps per episode (it lasts at most max_time_steps + 2).
    "max_time_steps": 8000,
    # Each decision is applied for 8 physics steps, so one decision equals
    # one PredPreyGrass grid step: ~1000 decisions per episode. A physics
    # step with a birth ends its decision early.
    "action_repeat": 8,
    # Initial populations (PredPreyGrass base_environment_discrete: 6 and 8).
    "predator_count": 6,
    "prey_count": 8,
    # Movement, in arena units per physics step. Prey are a little faster.
    "predator_mass": 10,
    "prey_mass": 3,
    "predator_radius": 16,  # A catch: centres closer than the sum of radii.
    "prey_radius": 16,
    "predator_max_velocity": 5,  # Top speed.
    "prey_max_velocity": 6.0,
    "predator_max_acceleration": 0.6,  # Speed gained per step.
    "prey_max_acceleration": 1,
    "predator_max_steer_force": 0.6,  # How sharply it can turn.
    "prey_max_steer_force": 0.6,
    # View cones (radius in arena units, angle in degrees).
    "predator_view_distance": 200,
    "prey_view_distance": 267,
    "predator_fov": 150,
    "prey_fov": 240,
    # Observation slots for the nearest predators and prey in view.
    "predator_observe_count": 1,
    "prey_observe_count": 3,
    # Grass: 100 patches in 20 clusters of 5 that move as prey graze them
    # (seed dispersal) and collapse when grazed bare (overgrazing).
    "grass_count": 100,
    "grass_consume_radius": 12.0,
    "grass_respawn_delay": 400,  # Physics steps until an eaten patch regrows.
    "grass_clustered": True,
    "grass_cluster_count": 20,
    "grass_cluster_spread": 24.0,
    "grass_random_respawn": False,
    "grass_dispersal": True,
    "grass_dispersal_distance": 24.0,
    "grass_overgrazing": True,
    "grass_overgrazing_threshold": 0.0,
    "grass_overgrazing_delay": 400,
    # Energy, in PredPreyGrass base_environment_discrete units per physics step (its
    # per-step rates divided by 8). 75% of the cost at full speed comes from
    # movement, linear in speed (Taylor, Heglund & Maloiy 1982).
    "energy_predator_initial": 5.0,
    "energy_prey_initial": 3.0,
    "energy_predator_max": 24.0,
    "energy_prey_max": 16.0,
    "energy_predator_resting_metabolic_cost": 0.25 * 0.15 / 8,
    "energy_prey_resting_metabolic_cost": 0.25 * 0.05 / 8,
    "energy_predator_speed_cost": 0.75 * 0.15 / 8 / 5,
    "energy_prey_speed_cost": 0.75 * 0.05 / 8 / 5,
    "energy_grass_gain": 2.0,
    # Share of a caught prey's energy the catching predator gains.
    "energy_catch_efficiency_predator": 0.5,
    # Reproduction: at the threshold an animal has one offspring with the
    # initial energy, paid by the parent, and is rewarded (the only reward).
    "reproduction_predator_threshold": 12.0,
    "reproduction_prey_threshold": 8.0,
    "reproduction_predator_reward": 10.0,
    "reproduction_prey_reward": 10.0,
    "reproduction_max_predators": 50,
    "reproduction_max_prey": 100,
    # Agent IDs per species, never reused within an episode.
    "reproduction_predator_pool": 2000,
    "reproduction_prey_pool": 2000,
    # Fixes of Aquarium's quirks (SPEC.md, section 9). All False reproduces
    # add_walls exactly; environment.FIXES turns them all on.
    "fix_simultaneous_movement": False,  # Everyone moves on the same positions.
    "fix_caught_prey_does_not_move": False,  # A caught prey is removed first.
    "fix_keep_heading_when_stopped": False,  # The view cone stays put.
    "fix_wrap_borders": False,  # Crossing an edge keeps the overshoot.
    "fix_same_collisions_for_both": False,  # Prey bumps add to acceleration too.
    "fix_all_contacts": False,  # A bump pushes away from every animal touched.
    "fix_episode_length": False,  # Exactly max_time_steps physics steps.
    "fix_no_spare_spawn_draw": False,  # No unused random draw per initial prey.
}
