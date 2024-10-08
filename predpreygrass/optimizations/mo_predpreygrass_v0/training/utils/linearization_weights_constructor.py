def construct_linearalized_weights(num_predators, num_prey):
    weights = {}
    # Populate the weights dictionary for predators
    for i in range(num_predators):
        weights[f"predator_{i}"] = [0.5, 0.5]
    # Populate the weights dictionary for prey
    for i in range(num_prey):
        weights[f"prey_{i + num_predators}"] = [0.5, 0.5]
    return weights
