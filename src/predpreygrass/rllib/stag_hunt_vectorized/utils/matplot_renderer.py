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
            if "type_1_predator" in agent:
                color = "#ff9999"  # light red
            elif "type_2_predator" in agent:
                color = "#cc0000"  # dark red
            elif "type_1_prey" in agent:
                color = "#9999ff"  # light blue
            elif "type_2_prey" in agent:
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
        self.type_counts_dict = {
            "type_1_predator": [],
            "type_2_predator": [],
            "type_1_prey": [],
            "type_2_prey": [],
        }

    def record_counts(self, active_agent_names):
        s1_pred = sum(1 for name in active_agent_names if "type_1_predator" in name)
        s2_pred = sum(1 for name in active_agent_names if "type_2_predator" in name)
        s1_prey = sum(1 for name in active_agent_names if "type_1_prey" in name)
        s2_prey = sum(1 for name in active_agent_names if "type_2_prey" in name)

        self.type_counts_dict["type_1_predator"].append(s1_pred)
        self.type_counts_dict["type_2_predator"].append(s2_pred)
        self.type_counts_dict["type_1_prey"].append(s1_prey)
        self.type_counts_dict["type_2_prey"].append(s2_prey)

    def plot(self):
        type_counts_dict = self.type_counts_dict
        plt.figure(figsize=(14, 6))

        # First subplot: Absolute counts
        plt.subplot(1, 2, 1)
        for label, counts in type_counts_dict.items():
            plt.plot(counts, label=label.replace("_", " ").capitalize())
        plt.xlabel("Step")
        plt.ylabel("Number of Agents")
        plt.title("Agent Population by type (Absolute Count)")
        plt.legend()
        plt.grid(True)

        # Second subplot: Proportions of type_2 agents only
        plt.subplot(1, 2, 2)
        total_steps = len(next(iter(type_counts_dict.values())))
        type_2_predator_props = []
        type_2_prey_props = []

        for step in range(total_steps):
            total_pred = sum(type_counts_dict[k][step] for k in type_counts_dict if "predator" in k)
            total_prey = sum(type_counts_dict[k][step] for k in type_counts_dict if "prey" in k)

            s2_pred = type_counts_dict.get("type_2_predator", [0] * total_steps)[step]
            s2_prey = type_counts_dict.get("type_2_prey", [0] * total_steps)[step]

            type_2_predator_props.append(s2_pred / total_pred if total_pred > 0 else 0)
            type_2_prey_props.append(s2_prey / total_prey if total_prey > 0 else 0)

        plt.plot(type_2_predator_props, label="High-type Predator Proportion")
        plt.plot(type_2_prey_props, label="High-type Prey Proportion")

        plt.xlabel("Step")
        plt.ylabel("Proportion")
        plt.title("Proportion of High-type Agents")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class AverageAgeVisualizer:
    def __init__(self):
        self.history = {
            "type_1_prey": [],
            "type_2_prey": [],
            "type_1_predator": [],
            "type_2_predator": [],
        }

    def record(self, agent_internal_ids, agent_ages, agent_positions):
        # Initialize sum and count per group
        sums = {k: 0 for k in self.history}
        counts = {k: 0 for k in self.history}

        for agent_id, internal_id in agent_internal_ids.items():
            if agent_id not in agent_positions:
                continue  # Only include active agents

            age = agent_ages.get(internal_id, 0)

            if "type_1_prey" in agent_id:
                sums["type_1_prey"] += age
                counts["type_1_prey"] += 1
            elif "type_2_prey" in agent_id:
                sums["type_2_prey"] += age
                counts["type_2_prey"] += 1
            elif "type_1_predator" in agent_id:
                sums["type_1_predator"] += age
                counts["type_1_predator"] += 1
            elif "type_2_predator" in agent_id:
                sums["type_2_predator"] += age
                counts["type_2_predator"] += 1

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
    def __init__(self, destination_path=None, timestamp=None, destination_filename="summary_plots", run_nr=None):
        self.destination_path = destination_path
        self.destination_filename = destination_filename
        self.timestamp = timestamp
        self.run_nr = run_nr
        # Population counts
        self.time_steps = []
        self.predator_counts = []
        self.prey_counts = []
        # type-based counts
        self.type_counts_dict = {"type_1_predator": [], "type_2_predator": [], "type_1_prey": [], "type_2_prey": []}
        # Energy by type series
        self.energy_by_type_series = []  # e.g., [{"type_1_prey": 120, "grass": 30, ...}, ...]

        # Age tracking
        self.average_ages = {"type_1_predator": [], "type_2_predator": [], "type_1_prey": [], "type_2_prey": []}

    def record(self, agent_ids):
        step = len(self.time_steps)
        self.time_steps.append(step)
        self.predator_counts.append(sum(1 for a in agent_ids if "predator" in a))
        self.prey_counts.append(sum(1 for a in agent_ids if "prey" in a))

        # type counts
        count_dict = {k: 0 for k in self.type_counts_dict}
        for agent_id in agent_ids:
            for group in count_dict:
                if group in agent_id:
                    count_dict[group] += 1
        for k in self.type_counts_dict:
            self.type_counts_dict[k].append(count_dict[k])

    def record_energy(self, energy_dict):
        """Stores a dict like {"type_1_prey": total_energy, ..., "grass": total_energy} for each step."""
        self.energy_by_type_series.append(energy_dict.copy())

    def plot(self):
        steps = self.time_steps
        n_charts = 2
        if self.energy_by_type_series:
            n_charts += 1
        plt.figure(figsize=(24, 6))
        color_map = {"type_1_predator": "#ff9999", "type_2_predator": "red", "type_1_prey": "#9999ff", "type_2_prey": "blue"}

        # 1. Total predator and prey count
        plt.subplot(1, n_charts, 1)
        plt.plot(steps, self.predator_counts, label="Predators", color="red", linewidth=2)
        plt.plot(steps, self.prey_counts, label="Prey", color="blue", linewidth=2)
        plt.title("Agent Population by Type")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        # 2. type-specific counts
        plt.subplot(1, n_charts, 2)
        for group, counts in self.type_counts_dict.items():
            plt.plot(steps, counts, label=group, color=color_map.get(group, "black"), linewidth=2)
        plt.title("Agent Population by type Type")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)

        # 3. Aggregate Energy by Type
        if self.energy_by_type_series:
            plt.subplot(1, n_charts, 3)
            steps = list(range(len(self.energy_by_type_series)))
            energy_keys = ["predator", "prey", "grass"]
            color_map = {"predator": "red", "prey": "blue", "grass": "green", "total": "black"}
            for k in energy_keys:
                series = [entry[k] for entry in self.energy_by_type_series]
                plt.plot(steps, series, label=k.capitalize(), linewidth=2, color=color_map[k])
            # Add total energy line
            total_series = [sum(entry[k] for k in energy_keys) for entry in self.energy_by_type_series]
            plt.plot(steps, total_series, label="Total", linestyle="--", linewidth=2, color=color_map["total"])

            plt.title("Total Energy per Agent Type")
            plt.xlabel("Step")
            plt.ylabel("Energy")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if self.destination_path:
            os.makedirs(os.path.join(self.destination_path, "summary_plots"), exist_ok=True)
            path = os.path.join(self.destination_path, "summary_plots", "evolution_summary_" + str(self.run_nr) + ".png")
            plt.savefig(path)
            # plt.show()
        else:
            plt.show()


class PreyDeathCauseVisualizer:
    def __init__(self, destination_path=None, timestamp=None, destination_filename="summary_plots"):
        self.timestamp = timestamp
        self.destination_path = destination_path
        self.destination_filename = destination_filename
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
            os.makedirs(os.path.join(self.destination_path, self.destination_filename), exist_ok=True)
            path = os.path.join(
                self.destination_path, self.destination_filename, "prey_death_cause_plot_" + str(self.timestamp) + ".png"
            )
            plt.savefig(path)
            plt.show()
        else:
            plt.show()
