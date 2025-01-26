import matplotlib.pyplot as plt
import numpy as np


class GridVisualizer:
    """
    A class for visualizing a grid-based environment using Matplotlib.
    """

    def __init__(self, grid_size, agents, trace_length=5):
        """
        Initialize the visualizer.

        Args:
            grid_size (tuple): Size of the grid (rows, cols).
            agents (list): List of agent names (e.g., ["predator_0", "prey_0"]).
            trace_length (int): Number of steps to retain the movement trace.
        """
        self.grid_size = grid_size
        self.agents = agents
        self.trace_length = trace_length

        # Initialize trace storage for agents
        self.agent_traces = {agent: [] for agent in agents}

        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(-0.5, grid_size[1] - 0.5)
        self.ax.set_ylim(-0.5, grid_size[0] - 0.5)
        self.ax.set_xticks(range(grid_size[1]))
        self.ax.set_yticks(range(grid_size[0]))
        self.ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
        self.ax.set_aspect("equal")

        # Agent markers
        self.predator_marker = "●"
        self.prey_marker = "◆"
        self.grass_marker = "■"

    def update(self, agent_positions):
        """
        Update the visualization with new agent positions.

        Args:
            agent_positions (dict): Dictionary of agent positions, e.g.,
                                    {"predator_0": [2, 3], "prey_0": [4, 5]}.
        """
        self.ax.clear()

        # Redraw the grid
        self.ax.set_xlim(-0.5, self.grid_size[1] - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size[0] - 0.5)
        self.ax.set_xticks(range(self.grid_size[1]))
        self.ax.set_yticks(range(self.grid_size[0]))
        self.ax.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
        self.ax.set_aspect("equal")

        # Update traces for each agent
        for agent, position in agent_positions.items():
            x, y = position

            # Skip trace updates for grass agents
            if "grass" in agent:
                # Render grass as a static marker
                self.ax.text(
                    y, x, self.grass_marker, color="green", fontsize=10, ha="center", va="center"
                )
                continue

            # Update movement traces for predators and prey
            if len(self.agent_traces[agent]) >= self.trace_length:
                self.agent_traces[agent].pop(0)
            self.agent_traces[agent].append(position)

        # Draw traces
        for agent, trace in self.agent_traces.items():
            trace_array = np.array(trace)
            if len(trace_array) > 1:
                self.ax.plot(
                    trace_array[:, 1],  # Y-coordinates
                    trace_array[:, 0],  # X-coordinates
                    color="red" if "predator" in agent else "blue",
                    alpha=0.6,
                    linewidth=1,
                    linestyle="-",
                )

        # Draw agents
        for agent, (x, y) in agent_positions.items():
            if "grass" in agent:
                continue  # Grass agents are static and already drawn
            marker = self.predator_marker if "predator" in agent else self.prey_marker
            color = "red" if "predator" in agent else "blue"
            self.ax.text(
                y, x, marker, color=color, fontsize=12, ha="center", va="center"
            )

        # Redraw the plot
        plt.pause(0.0000001)

    def close(self):
        """
        Close the visualization.
        """
        plt.close(self.fig)
