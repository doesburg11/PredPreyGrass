import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class PopulationPlotter:
    def __init__(
        self,
        n_active_predator_list,
        n_active_prey_list,
        n_aec_cycles,
        episode_number,
        output_directory,
        
    ):
        self.n_active_predator_list = n_active_predator_list
        self.n_active_prey_list = n_active_prey_list
        self.n_aec_cycles = n_aec_cycles
        self.episode_number = episode_number
        self.output_directory = output_directory

    def plot(self, title="Predator and Prey Population Over Time"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.n_active_predator_list, label="Predator Population", color="red")
        plt.plot(self.n_active_prey_list, label="Prey Population", color="green")
        plt.xlabel("Time Steps")
        plt.ylabel("Population")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xlim([0, self.n_aec_cycles])
        plt.ylim(
            [0, max(max(self.n_active_predator_list), max(self.n_active_prey_list))]
        )
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
