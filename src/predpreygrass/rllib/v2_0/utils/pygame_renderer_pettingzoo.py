import os
import pygame
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


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

        self.save_image_steps = False  # TODO put into config file?

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
                cell_rect = pygame.Rect(self.cell_scale * x, self.cell_scale * y, self.cell_scale, self.cell_scale)
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
                self.screen.blit(patch, (self.cell_scale * (pos_x - offset + 0.5), self.cell_scale * (pos_y - offset + 0.5)))

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
            center = (int(self.cell_scale * x + self.cell_scale / 2), int(self.cell_scale * y + self.cell_scale / 2))
            pygame.draw.circle(self.screen, color, center, int(self.cell_scale / 2.3))

    def _draw_energy_title(self, x_pos, y_pos):
        title_font = pygame.font.Font(None, 40)
        title = title_font.render("Energy Level Agents", True, (0, 0, 0))
        self.screen.blit(title, (x_pos, y_pos))

    def _draw_energy_label_predator(self, x_pos, y_pos):
        predator_label_font = pygame.font.Font(None, 35)
        predator_label = predator_label_font.render("Predators", True, (255, 0, 0))
        self.screen.blit(predator_label, (x_pos, self.y_position_predator_chart + 180))

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
            self.screen = pygame.Surface((self.cell_scale * self.env.x_grid_size, self.cell_scale * self.env.y_grid_size))

    def _draw_agent_ids(self):
        font = pygame.font.SysFont("Comic Sans MS", self.cell_scale * 2 // 3)
        for agent_list in [
            self.env.active_agent_instance_list_type[self.env.predator_type_nr],
            self.env.active_agent_instance_list_type[self.env.prey_type_nr],
            self.env.active_agent_instance_list_type[self.env.grass_type_nr],
        ]:
            for instance in agent_list:
                x, y = instance.position
                pos_x, pos_y = self.cell_scale * x + self.cell_scale // 6, self.cell_scale * y + self.cell_scale // 1.2
                text = font.render(str(instance.agent_id_nr), False, (255, 255, 0))
                self.screen.blit(text, (pos_x, pos_y - self.cell_scale // 2))

    def _draw_energy_chart_predators(self, offset_x):
        x_pos, y_pos = self.cell_scale * self.env.x_grid_size + offset_x, 0
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_title(x_pos + 750, 20)
        self._draw_energy_bars_predators(x_pos + 100, 50, self.width_energy_chart - 100, 400)
        self._draw_energy_label_predator(x_pos + 100, 585)

    def _draw_energy_chart_prey(self, offset_x):
        x_pos, y_pos = self.cell_scale * self.env.x_grid_size + offset_x, 500
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_title(x_pos + 750, 20)
        self._draw_energy_bars_prey(
            x_pos + 100, 100 + self.y_position_prey_chart, self.width_energy_chart - 100, self.y_position_prey_chart
        )
        self._draw_energy_label_prey(x_pos + 100, self.y_position_prey_chart + 550)

    def _draw_energy_bars_predators(self, x_pos, y_pos, width, height):
        bar_width, offset, max_energy, red = 10, 14, 50, (255, 0, 0)
        y_axis_x = x_pos - 20
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
        bar_width, offset, max_energy, blue = 10, 14, 50, (0, 0, 255)
        y_axis_x = x_pos - 20
        x_axis_y = y_pos + height

        # Draw Axes
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, y_pos + 0, 5, height))
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, x_axis_y + 0, width - 0, 5))

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
