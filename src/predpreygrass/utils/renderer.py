import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

class PyGameRenderer:
    """
    A class for visualizing a grid-based environment using PyGame. Is used in the PettingZoo framework.
    """
    def __init__(self, env, cell_scale=40, has_energy_chart=True, x_pygame_window=0, y_pygame_window=0):
        self.env = env
        self.cell_scale = cell_scale
        self.has_energy_chart = has_energy_chart
        self.file_name = 0

        # Initialize the Pygame window position
        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0" 

        # Pygame screen settings
        self.width = env.x_grid_size * self.cell_scale
        self.height = env.y_grid_size * self.cell_scale
        self.width_energy_chart = 1560 if has_energy_chart else 0
        self.height_energy_chart: int = self.cell_scale * self.env.y_grid_size

        self.save_image_steps = False # TODO put into config file? 

        # BAR CHART POSITION
        self.y_position_predator_chart = 300
        self.y_position_prey_chart = 420

        pygame.init()
        if env.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width + self.width_energy_chart, self.height))
            pygame.display.set_caption("PredPreyGrass - MARL Environment")
        else:
            self.screen = pygame.Surface((self.width, self.height))

    def render(self):
        if self.screen is None:
            return

        self._draw_grid()
        self._draw_observations(self.env.active_agent_instance_list_type[self.env.prey_type_nr], (72, 152, 255))
        self._draw_observations(self.env.active_agent_instance_list_type[self.env.predator_type_nr], (255, 152, 72))
        self._draw_instances(self.env.active_agent_instance_list_type[self.env.grass_type_nr], (0, 128, 0))
        self._draw_instances(self.env.active_agent_instance_list_type[self.env.prey_type_nr], (0, 0, 255))
        self._draw_instances(self.env.active_agent_instance_list_type[self.env.predator_type_nr], (255, 0, 0))
        self._draw_agent_ids()

        if self.has_energy_chart:
            self._draw_energy_chart_predators(0)
            self._draw_energy_chart_prey(0)

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation

        if self.env.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            if self.save_image_steps:
                self._save_image()

        return np.transpose(new_observation, axes=(1, 0, 2)) if self.env.render_mode == "rgb_array" else None

    def _draw_grid(self):
        for x in range(self.env.x_grid_size):
            for y in range(self.env.y_grid_size):
                cell_rect = pygame.Rect(
                    self.cell_scale * x, self.cell_scale * y, self.cell_scale, self.cell_scale
                )
                pygame.draw.rect(self.screen, (255, 255, 255), cell_rect)
                pygame.draw.rect(self.screen, (192, 192, 192), cell_rect, 1)
        border_rect = pygame.Rect(0, 0, self.cell_scale * self.env.x_grid_size, self.cell_scale * self.env.y_grid_size)
        pygame.draw.rect(self.screen, (255, 0, 0), border_rect, 5)

    def _draw_observations(self, instances, color):
        for instance in instances:
            x, y = instance.position
            size = self.cell_scale * instance.observation_range
            patch = pygame.Surface((size, size))
            patch.set_alpha(128)
            patch.fill(color)
            offset = instance.observation_range / 2.0

            # Handle is_torus wrapping
            positions = [(x, y)]
            if self.env.is_torus:  # If the environment has is_torus topology
                positions = self._get_torus_positions(x, y, instance.observation_range)

            for pos_x, pos_y in positions:
                self.screen.blit(
                    patch,
                    (self.cell_scale * (pos_x - offset + 0.5), self.cell_scale * (pos_y - offset + 0.5))
                )

    def _get_torus_positions(self, x, y, observation_range):
        """
        Get all possible positions for an observation patch in a toroidal environment.
        """
        positions = [(x, y)]
        offset = observation_range // 2

        if x - offset < 0:
            positions.append((x + self.env.x_grid_size, y))
        if x + offset >= self.env.x_grid_size:
            positions.append((x - self.env.x_grid_size, y))
        if y - offset < 0:
            positions.append((x, y + self.env.y_grid_size))
        if y + offset >= self.env.y_grid_size:
            positions.append((x, y - self.env.y_grid_size))

        # Handle corners for is_torus wrapping
        if x - offset < 0 and y - offset < 0:
            positions.append((x + self.env.x_grid_size, y + self.env.y_grid_size))
        if x - offset < 0 and y + offset >= self.env.y_grid_size:
            positions.append((x + self.env.x_grid_size, y - self.env.y_grid_size))
        if x + offset >= self.env.x_grid_size and y - offset < 0:
            positions.append((x - self.env.x_grid_size, y + self.env.y_grid_size))
        if x + offset >= self.env.x_grid_size and y + offset >= self.env.y_grid_size:
            positions.append((x - self.env.x_grid_size, y - self.env.y_grid_size))

        return positions

    def _draw_instances(self, instances, color):
        for instance in instances:
            x, y = instance.position
            center = (
                int(self.cell_scale * x + self.cell_scale / 2),
                int(self.cell_scale * y + self.cell_scale / 2)
            )
            pygame.draw.circle(self.screen, color, center, int(self.cell_scale / 2.3))

    def _draw_energy_title(self, x_pos, y_pos):
        title_font = pygame.font.Font(None, 40)
        title = title_font.render("Energy Level Agents", True, (0, 0, 0))
        self.screen.blit(title, (x_pos, y_pos))

    def _draw_energy_label_predator(self, x_pos, y_pos):
        predator_label_font = pygame.font.Font(None, 35)
        predator_label = predator_label_font.render("Predators", True, (255, 0, 0))
        self.screen.blit(predator_label, (x_pos, self.y_position_predator_chart+ 180))

    def _draw_energy_label_prey(self, x_pos, y_pos):
        prey_label_font = pygame.font.Font(None, 35)
        prey_label = prey_label_font.render("Prey", True, (0, 0, 255))
        self.screen.blit(prey_label, (x_pos, y_pos))

    def _initialize_screen(self):
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.cell_scale * self.env.x_grid_size + self.width_energy_chart, self.cell_scale * self.env.y_grid_size)
            )
            pygame.display.set_caption("PredPreyGrass - create agents")
        else:
            self.screen = pygame.Surface(
                (self.cell_scale * self.env.x_grid_size, self.cell_scale * self.env.y_grid_size)
            )

    def _draw_agent_ids(self):
        font = pygame.font.SysFont("Comic Sans MS", self.cell_scale * 2 // 3)
        for agent_list in [self.env.active_agent_instance_list_type[self.env.predator_type_nr], self.env.active_agent_instance_list_type[self.env.prey_type_nr], self.env.active_agent_instance_list_type[self.env.grass_type_nr]]:
            for instance in agent_list:
                x, y = instance.position
                pos_x, pos_y = self.cell_scale * x + self.cell_scale // 6, self.cell_scale * y + self.cell_scale // 1.2
                text = font.render(str(instance.agent_id_nr), False, (255, 255, 0))
                self.screen.blit(text, (pos_x, pos_y - self.cell_scale // 2))

    def _draw_energy_chart_predators(self, offset_x):
        x_pos, y_pos = self.cell_scale * self.env.x_grid_size + offset_x, 0
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_title(x_pos + 750, 20)
        self._draw_energy_bars_predators(
            x_pos + 100, 
            50, 
            self.width_energy_chart - 100, 
            400)
        self._draw_energy_label_predator(x_pos + 100, 585)

    def _draw_energy_chart_prey(self, offset_x):
        x_pos, y_pos = self.cell_scale * self.env.x_grid_size + offset_x, 500
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_title(x_pos + 750, 20)
        self._draw_energy_bars_prey(
            x_pos + 100, 
            100+self.y_position_prey_chart, 
            self.width_energy_chart - 100, 
            self.y_position_prey_chart
            )
        self._draw_energy_label_prey(x_pos + 100, self.y_position_prey_chart+550)

    def _draw_energy_title(self, x_pos, y_pos):
        title_font = pygame.font.Font(None, 40)
        title = title_font.render("Energy Level Agents", True, (0, 0, 0))
        self.screen.blit(title, (x_pos, y_pos))

    def _draw_energy_bars_predators(self, x_pos, y_pos, width, height):
        bar_width, offset, max_energy, red, blue = 10, 14, 50, (255, 0, 0), (0, 0, 255)
        y_axis_x = x_pos -20
        x_axis_y = y_pos + height 

        # Draw Axes
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, y_pos, 5, height))
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, x_axis_y, width - 490, 5))

        # Draw Bars and Labels for Predators
        for i, name in enumerate(self.env.possible_agent_name_list_type[self.env.predator_type_nr]):
            instance = self.env.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            bar_x, bar_y = x_pos + i * (bar_width + offset), y_pos + height - bar_height
            pygame.draw.rect(self.screen, red, (bar_x, bar_y, bar_width, bar_height))
            label_x, label_y = bar_x, x_axis_y + 10
            label = pygame.font.Font(None, 20).render(str(instance.agent_id_nr), True, red)
            self.screen.blit(label, (label_x, label_y))

        # Draw Tick Points on Y-Axis
        for i in range(max_energy + 1):
            if i % 5 == 0:
                tick_y = y_pos + height - (i / max_energy) * height
                pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x - 5, tick_y, 10, 2))
                label = pygame.font.Font(None, 30).render(str(i), True, (0, 0, 0))
                self.screen.blit(label, (y_axis_x - 35, tick_y - 5))

    def _draw_energy_bars_prey(self, x_pos, y_pos, width, height):
        bar_width, offset, max_energy, red, blue = 10, 14, 50, (255, 0, 0), (0, 0, 255)
        y_axis_x = x_pos -20
        x_axis_y = y_pos + height 

        # Draw Axes
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, y_pos+0, 5, height))
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, x_axis_y+0, width - 0, 5))

        # Draw Bars and Labels for Prey
        for i, name in enumerate(self.env.possible_agent_name_list_type[self.env.prey_type_nr]):
            instance = self.env.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            bar_x, bar_y = x_pos + i * (bar_width + offset), y_pos + height - bar_height
            pygame.draw.rect(self.screen, blue, (bar_x, bar_y, bar_width, bar_height))
            label_x, label_y = bar_x, x_axis_y + 10
            label = pygame.font.Font(None, 20).render(str(instance.agent_id_nr), True, blue)
            self.screen.blit(label, (label_x, label_y))

        # Draw Tick Points on Y-Axis
        for i in range(max_energy + 1):
            if i % 5 == 0:
                tick_y = y_pos + height - (i / max_energy) * height
                pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x - 5, tick_y, 10, 2))
                label = pygame.font.Font(None, 30).render(str(i), True, (0, 0, 0))
                self.screen.blit(label, (y_axis_x - 35, tick_y - 5))

    def _save_image(self):
        self.file_name += 1
        # print(f"{self.file_name}.png saved")
        pygame.image.save(self.screen, f"./assets/images/{self.file_name}.png")

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

class MatPlotLibRenderer:
    """
    A class for visualizing a grid-based environment using Matplotlib. Is used in the RLLib framework.
    """

    #def __init__(self, grid_size, agents, trace_length=5):
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
                cell_size = min(self.fig.get_figwidth() * self.fig.dpi / self.grid_size[1],
                                self.fig.get_figheight() * self.fig.dpi / self.grid_size[0])
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
            cell_size = min(self.fig.get_figwidth() * self.fig.dpi / self.grid_size[1],
                            self.fig.get_figheight() * self.fig.dpi / self.grid_size[0])
            font_size = cell_size * 0.6  # 60% of the cell size for padding

            self.agent_texts[agent] = self.ax.text(
                y, x, marker, color=color, fontsize=font_size, ha="center", va="center"
            )

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
            'speed_1_predator': [],
            'speed_2_predator': [],
            'speed_1_prey': [],
            'speed_2_prey': [],
        }

    def record_counts(self, active_agent_names):
        s1_pred = sum(1 for name in active_agent_names if "speed_1_predator" in name)
        s2_pred = sum(1 for name in active_agent_names if "speed_2_predator" in name)
        s1_prey = sum(1 for name in active_agent_names if "speed_1_prey" in name)
        s2_prey = sum(1 for name in active_agent_names if "speed_2_prey" in name)

        self.speed_counts_dict['speed_1_predator'].append(s1_pred)
        self.speed_counts_dict['speed_2_predator'].append(s2_pred)
        self.speed_counts_dict['speed_1_prey'].append(s1_prey)
        self.speed_counts_dict['speed_2_prey'].append(s2_prey)

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

            s2_pred = speed_counts_dict.get("speed_2_predator", [0]*total_steps)[step]
            s2_prey = speed_counts_dict.get("speed_2_prey", [0]*total_steps)[step]

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
        plt.plot(self.time_steps, self.predator_counts, label='Predators', color='red')
        plt.plot(self.time_steps, self.prey_counts, label='Prey', color='blue')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Agents')
        plt.title('Agent Population Over Time')
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
    def __init__(self, destination_path=None):
        self.destination_path = destination_path

        # Population counts
        self.time_steps = []
        self.predator_counts = []
        self.prey_counts = []

        # Speed-based counts
        self.speed_counts_dict = {
            "speed_1_predator": [],
            "speed_2_predator": [],
            "speed_1_prey": [],
            "speed_2_prey": []
        }

        # Age tracking
        self.average_ages = {
            "speed_1_predator": [],
            "speed_2_predator": [],
            "speed_1_prey": [],
            "speed_2_prey": []
        }

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
        color_map = {
            'speed_1_predator': "#ff9999",
            'speed_2_predator': "red",
            'speed_1_prey': "#9999ff",
            'speed_2_prey': "blue"
        }

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
            path = os.path.join(self.destination_path, "summary_plots", "evolution_summary.png")
            plt.savefig(path)
            plt.show()
        else:
            plt.show()

class PreyDeathCauseVisualizer:
    def __init__(self, destination_path=None):
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
        #plt.plot(self.time_steps, [e * 100 for e in self.eaten_ratio], label="Eaten Prey %", color="black", linestyle="--", linewidth=2)
        plt.title("Prey Death Cause Starvation Relative")
        plt.xlabel("Step")
        plt.ylabel("Percentage (%)")
        #plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)

        if self.destination_path:
            os.makedirs(os.path.join(self.destination_path, "summary_plots"), exist_ok=True)
            path = os.path.join(self.destination_path, "summary_plots", "prey_death_cause_plot.png")
            plt.savefig(path)
            plt.show()
        else:
            plt.show()
