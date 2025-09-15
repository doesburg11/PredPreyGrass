import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


class MatPlotLibRenderer:
    """
    A class for visualizing a grid-based environment using Matplotlib. Is used in the RLLib framework.
    """

    # def __init__(self, grid_size, agents, trace_length=5):
    def __init__(self, grid_size, agents, trace_length=5, show_gridlines=True, scale=1.0, destination_path=None):
        """
        Initialize the visualizer.

        Args:
            grid_size (tuple): Size of the grid (rows, cols).
            agents (list): List of agent names (e.g., ["predator_0", "prey_0"]).
            trace_length (int): Number of steps to retain the movement trace.
            destination_path (str): Directory to save plots. If None, plots are not saved automatically.
        """
        self.grid_size = grid_size
        self.agents = set(agents)
        self.trace_length = trace_length
        self.agent_traces = {agent: [] for agent in agents}
        self.show_gridlines = show_gridlines
        self.scale = scale
        self.destination_path = destination_path

        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(6 * self.scale, 6 * self.scale))
        self.ax.set_xlim(-0.5, grid_size[1] - 0.5)
        self.ax.set_ylim(-0.5, grid_size[0] - 0.5)

        if self.show_gridlines:
            self.ax.set_xticks(range(grid_size[1]))
            self.ax.set_yticks(range(grid_size[0]))
            self.ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.grid(False)

        self.ax.set_aspect("equal")

        # Flip the y-axis so (0,0) is at the bottom-left
        self.ax.invert_yaxis()

        # Store agent markers
        self.agent_texts = {}

        # Store trace lines as Line2D objects
        self.trace_lines = {}
        for agent in agents:
            color = "red" if "predator" in agent else "blue"
            self.trace_lines[agent] = Line2D([], [], color=color, alpha=0.6, linewidth=1, linestyle="-")
            self.ax.add_line(self.trace_lines[agent])

        # Agent markers
        self.predator_marker = "●"
        self.prey_marker = "◆"
        self.grass_marker = "■"

    def update(self, agent_positions, step):
        self.ax.set_title(f"Environment - Step {step}", fontsize=14)

        # Remove old agent markers
        for text in self.agent_texts.values():
            text.remove()
        self.agent_texts.clear()

        # Get current agent names
        current_agents = set(agent_positions.keys())

        # Remove traces of dead agents
        dead_agents = self.agents - current_agents
        for agent in dead_agents:
            if agent in self.trace_lines:
                self.trace_lines[agent].set_data([], [])
            if agent in self.agent_traces:
                del self.agent_traces[agent]
        self.agents = current_agents

        # Update traces for remaining agents
        for agent, position in agent_positions.items():
            if "grass" in agent:
                # Use the same font scaling for grass
                cell_size = min(
                    self.fig.get_figwidth() * self.fig.dpi / self.grid_size[1],
                    self.fig.get_figheight() * self.fig.dpi / self.grid_size[0],
                )
                font_size = cell_size * 0.6

                self.agent_texts[agent] = self.ax.text(
                    position[1], position[0], self.grass_marker, color="green", fontsize=font_size, ha="center", va="center"
                )
                continue

            if agent not in self.agent_traces:
                self.agent_traces[agent] = []
            if len(self.agent_traces[agent]) >= self.trace_length:
                self.agent_traces[agent].pop(0)
            self.agent_traces[agent].append(position)

            trace_array = np.array(self.agent_traces[agent])
            if len(trace_array) > 1:
                self.trace_lines[agent].set_data(trace_array[:, 1], trace_array[:, 0])

        # Draw agents
        for agent, (x, y) in agent_positions.items():
            if "grass" in agent:
                continue

            # Determine marker
            marker = self.predator_marker if "predator" in agent else self.prey_marker

            # Determine color
            if "speed_1_predator" in agent:
                color = "#ff9999"  # light red
            elif "speed_2_predator" in agent:
                color = "#cc0000"  # dark red
            elif "speed_1_prey" in agent:
                color = "#9999ff"  # light blue
            elif "speed_2_prey" in agent:
                color = "#0000cc"  # dark blue
            elif "predator" in agent:
                color = "red"  # fallback for classic predator
            elif "prey" in agent:
                color = "blue"  # fallback for classic prey
            else:
                color = "black"  # unknown agent type fallback

            # Scale font to cell size
            cell_size = min(
                self.fig.get_figwidth() * self.fig.dpi / self.grid_size[1],
                self.fig.get_figheight() * self.fig.dpi / self.grid_size[0],
            )
            font_size = cell_size * 0.6  # 60% of the cell size for padding

            self.agent_texts[agent] = self.ax.text(y, x, marker, color=color, fontsize=font_size, ha="center", va="center")

        plt.draw()
        plt.pause(0.01)

    def save_frame(self, step, prefix="grid_step"):
        """
        Save the current grid as an image file with the step number in the filename.
        """
        if self.destination_path:
            save_dir = os.path.join(self.destination_path, "grid_step_plots")
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{prefix}_{step:04d}.png"
            filepath = os.path.join(save_dir, filename)
            self.fig.savefig(filepath)
            # print(f"Saved grid frame: {filepath}")

    def plot(self, save_name="grid_summary.png"):
        """Save the final plot summary."""
        if self.destination_path:
            os.makedirs(self.destination_path, exist_ok=True)
            path = os.path.join(self.destination_path, save_name)
            self.fig.savefig(path)
            # print(f"Saved final grid plot: {path}")
        else:
            # Display the population chart plot to the user
            plt.show()

    def close(self):
        """Close the visualization."""
        plt.close(self.fig)


class EvolutionVisualizer:
    def __init__(self):
        self.speed_counts_dict = {
            "speed_1_predator": [],
            "speed_2_predator": [],
            "speed_1_prey": [],
            "speed_2_prey": [],
        }

    def record_counts(self, active_agent_names):
        s1_pred = sum(1 for name in active_agent_names if "speed_1_predator" in name)
        s2_pred = sum(1 for name in active_agent_names if "speed_2_predator" in name)
        s1_prey = sum(1 for name in active_agent_names if "speed_1_prey" in name)
        s2_prey = sum(1 for name in active_agent_names if "speed_2_prey" in name)

        self.speed_counts_dict["speed_1_predator"].append(s1_pred)
        self.speed_counts_dict["speed_2_predator"].append(s2_pred)
        self.speed_counts_dict["speed_1_prey"].append(s1_prey)
        self.speed_counts_dict["speed_2_prey"].append(s2_prey)

    def plot(self):
        speed_counts_dict = self.speed_counts_dict
        plt.figure(figsize=(14, 6))

        # First subplot: Absolute counts
        plt.subplot(1, 2, 1)
        for label, counts in speed_counts_dict.items():
            plt.plot(counts, label=label.replace("_", " ").capitalize())
        plt.xlabel("Step")
        plt.ylabel("Number of Agents")
        plt.title("Agent Population by Speed (Absolute Count)")
        plt.legend()
        plt.grid(True)

        # Second subplot: Proportions of speed_2 agents only
        plt.subplot(1, 2, 2)
        total_steps = len(next(iter(speed_counts_dict.values())))
        speed_2_predator_props = []
        speed_2_prey_props = []

        for step in range(total_steps):
            total_pred = sum(speed_counts_dict[k][step] for k in speed_counts_dict if "predator" in k)
            total_prey = sum(speed_counts_dict[k][step] for k in speed_counts_dict if "prey" in k)

            s2_pred = speed_counts_dict.get("speed_2_predator", [0] * total_steps)[step]
            s2_prey = speed_counts_dict.get("speed_2_prey", [0] * total_steps)[step]

            speed_2_predator_props.append(s2_pred / total_pred if total_pred > 0 else 0)
            speed_2_prey_props.append(s2_prey / total_prey if total_prey > 0 else 0)

        plt.plot(speed_2_predator_props, label="High-Speed Predator Proportion")
        plt.plot(speed_2_prey_props, label="High-Speed Prey Proportion")

        plt.xlabel("Step")
        plt.ylabel("Proportion")
        plt.title("Proportion of High-Speed Agents")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class AverageAgeVisualizer:
    def __init__(self):
        self.history = {
            "speed_1_prey": [],
            "speed_2_prey": [],
            "speed_1_predator": [],
            "speed_2_predator": [],
        }

    def record(self, agent_internal_ids, agent_ages, agent_positions):
        # Initialize sum and count per group
        sums = {k: 0 for k in self.history}
        counts = {k: 0 for k in self.history}

        for agent_id, internal_id in agent_internal_ids.items():
            if agent_id not in agent_positions:
                continue  # Only include active agents

            age = agent_ages.get(internal_id, 0)

            if "speed_1_prey" in agent_id:
                sums["speed_1_prey"] += age
                counts["speed_1_prey"] += 1
            elif "speed_2_prey" in agent_id:
                sums["speed_2_prey"] += age
                counts["speed_2_prey"] += 1
            elif "speed_1_predator" in agent_id:
                sums["speed_1_predator"] += age
                counts["speed_1_predator"] += 1
            elif "speed_2_predator" in agent_id:
                sums["speed_2_predator"] += age
                counts["speed_2_predator"] += 1

        # Compute and record averages (0 if no agents)
        for group in self.history:
            avg = sums[group] / counts[group] if counts[group] > 0 else 0
            self.history[group].append(avg)

    def plot(self):
        plt.figure(figsize=(10, 5))
        for label, values in self.history.items():
            plt.plot(values, label=label.replace("_", " ").capitalize())
        plt.xlabel("Step")
        plt.ylabel("Average Age")
        plt.title("Average Age of Agent Groups Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class PopulationChart:
    def __init__(self, destination_path=None):
        self.destination_path = destination_path
        self.time_steps = []
        self.predator_counts = []
        self.prey_counts = []

    def record(self, step, agents):
        self.time_steps.append(step)
        self.predator_counts.append(sum(1 for a in agents if "predator" in a))
        self.prey_counts.append(sum(1 for a in agents if "prey" in a))

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_steps, self.predator_counts, label="Predators", color="red")
        plt.plot(self.time_steps, self.prey_counts, label="Prey", color="blue")
        plt.xlabel("Time Step")
        plt.ylabel("Number of Agents")
        plt.title("Agent Population Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.destination_path:
            plot_dir = os.path.join(self.destination_path, "summary_plots")
            os.makedirs(plot_dir, exist_ok=True)
            filepath = os.path.join(plot_dir, "population_chart.png")
            plt.savefig(filepath)
            # print(f"Population chart saved to: {filepath}")
        else:
            plt.show()


class CombinedEvolutionVisualizer:
    def __init__(self, destination_path=None, timestamp=None):
        self.destination_path = destination_path
        self.timestamp = timestamp

        # Population counts
        self.time_steps = []
        self.predator_counts = []
        self.prey_counts = []

        # Speed-based counts
        self.speed_counts_dict = {"speed_1_predator": [], "speed_2_predator": [], "speed_1_prey": [], "speed_2_prey": []}

        # Age tracking
        self.average_ages = {"speed_1_predator": [], "speed_2_predator": [], "speed_1_prey": [], "speed_2_prey": []}

    def record(self, agent_ids, internal_ids, agent_ages):
        step = len(self.time_steps)
        self.time_steps.append(step)
        self.predator_counts.append(sum(1 for a in agent_ids if "predator" in a))
        self.prey_counts.append(sum(1 for a in agent_ids if "prey" in a))

        # Speed counts
        count_dict = {k: 0 for k in self.speed_counts_dict}
        for agent_id in agent_ids:
            for group in count_dict:
                if group in agent_id:
                    count_dict[group] += 1
        for k in self.speed_counts_dict:
            self.speed_counts_dict[k].append(count_dict[k])

        # Average ages
        age_sums = {k: 0 for k in self.average_ages}
        age_counts = {k: 0 for k in self.average_ages}
        for agent_id in agent_ids:
            for group in self.average_ages:
                if group in agent_id:
                    internal_id = internal_ids[agent_id]
                    age_sums[group] += agent_ages[internal_id]
                    age_counts[group] += 1
        for group in self.average_ages:
            avg = age_sums[group] / age_counts[group] if age_counts[group] > 0 else 0
            self.average_ages[group].append(avg)

    def plot(self):
        steps = self.time_steps
        plt.figure(figsize=(24, 6))
        color_map = {"speed_1_predator": "#ff9999", "speed_2_predator": "red", "speed_1_prey": "#9999ff", "speed_2_prey": "blue"}

        # 1. Total predator and prey count
        plt.subplot(1, 4, 1)
        plt.plot(steps, self.predator_counts, label="Predators", color="red", linewidth=2)
        plt.plot(steps, self.prey_counts, label="Prey", color="blue", linewidth=2)
        plt.title("Agent Population by Type")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        # 2. Speed-specific counts
        plt.subplot(1, 4, 2)
        for group, counts in self.speed_counts_dict.items():
            label = group.replace("speed_1", "Low-Speed").replace("speed_2", "High-Speed").replace("_", " ").capitalize()
            plt.plot(steps, counts, label=label, color=color_map.get(group, "black"), linewidth=2)
        plt.title("Agent Population by Speed Type")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        # 3. High-speed proportions
        plt.subplot(1, 4, 3)
        predator_props, prey_props = [], []
        for i in steps:
            pred1 = self.speed_counts_dict["speed_1_predator"][i]
            pred2 = self.speed_counts_dict["speed_2_predator"][i]
            prey1 = self.speed_counts_dict["speed_1_prey"][i]
            prey2 = self.speed_counts_dict["speed_2_prey"][i]
            predator_props.append(pred2 / (pred1 + pred2) if (pred1 + pred2) > 0 else 0)
            prey_props.append(prey2 / (prey1 + prey2) if (prey1 + prey2) > 0 else 0)

        plt.plot(steps, [p * 100 for p in predator_props], label="High-Speed Predator %", color="#cc0000", linewidth=2)
        plt.plot(steps, [p * 100 for p in prey_props], label="High-Speed Prey %", color="#0000cc", linewidth=2)
        plt.title("High-Speed Agent Relative")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Step")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)

        # 4. Average ages
        plt.subplot(1, 4, 4)
        for group, ages in self.average_ages.items():
            label = group.replace("speed_1", "Low-Speed").replace("speed_2", "High-Speed").replace("_", " ").capitalize()
            plt.plot(steps, ages, label=label, color=color_map.get(group, "black"), linewidth=2)
        plt.title("Average Age per Agent Type")
        plt.xlabel("Step")
        plt.ylabel("Age")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if self.destination_path:
            os.makedirs(os.path.join(self.destination_path, "summary_plots"), exist_ok=True)
            path = os.path.join(self.destination_path, "summary_plots", "evolution_summary_" + str(self.timestamp) + ".png")
            plt.savefig(path)
            plt.show()
        else:
            plt.show()


class PreyDeathCauseVisualizer:
    def __init__(self, destination_path=None, timestamp=None):
        self.timestamp = timestamp
        self.destination_path = destination_path
        self.time_steps = []
        self.starved_ratio = []
        self.eaten_ratio = []

    def record(self, death_cause_prey):
        step = len(self.time_steps)
        self.time_steps.append(step)
        starved = sum(1 for cause in death_cause_prey.values() if cause == "starved")
        eaten = sum(1 for cause in death_cause_prey.values() if cause == "eaten")
        total = starved + eaten
        self.starved_ratio.append(starved / total if total > 0 else 0)
        self.eaten_ratio.append(eaten / total if total > 0 else 0)

    def plot(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.time_steps, [s * 100 for s in self.starved_ratio], label="Starved Prey %", color="blue", linewidth=2)
        # plt.plot(self.time_steps, [e * 100 for e in self.eaten_ratio], label="Eaten Prey %", color="black", linestyle="--", linewidth=2)
        plt.title("Prey Death Cause Starvation Relative")
        plt.xlabel("Step")
        plt.ylabel("Percentage (%)")
        # plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)

        if self.destination_path:
            os.makedirs(os.path.join(self.destination_path, "summary_plots"), exist_ok=True)
            path = os.path.join(self.destination_path, "summary_plots", "prey_death_cause_plot_" + str(self.timestamp) + ".png")
            plt.savefig(path)
            plt.show()
        else:
            plt.show()
