# population_plotter.py

import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


class PopulationPlotter:
    def __init__(self, output_directory):
        self.output_directory = output_directory

    def plot_population(
        self,
        n_active_predator_list,
        n_active_prey_list,
        n_aec_cycles,
        episode_number,
        title="Predator and Prey Population Over Time",
    ):
        plt.clf()
        plt.plot(n_active_predator_list, "r")
        plt.plot(n_active_prey_list, "b")
        plt.title(title, weight="bold")
        plt.xlabel("Time steps", weight="bold")
        ax = plt.gca()
        # Set x and y limits
        ax.set_xlim([0, n_aec_cycles])
        ax.set_ylim(
            [
                0,
                max(n_active_predator_list + n_active_prey_list),
            ]
        )
        # Display only whole numbers on the y-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # Remove box/spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # Make axes thicker
        plt.axhline(0, color="black", linewidth=4)
        plt.axvline(0, color="black", linewidth=4)
        # Make tick marks thicker
        plt.tick_params(width=2)

        population_dir = os.path.join(self.output_directory, "population_charts")
        os.makedirs(population_dir, exist_ok=True)
        model_file_name = os.path.join(
            population_dir, f"PredPreyPopulation_episode_{episode_number}.pdf"
        )
        plt.savefig(model_file_name)
