import os
import pygame
import numpy as np

class PyGameRenderer:
    def __init__(self, env, cell_scale=40, has_energy_chart=True,
                 grid_width=None, grid_height=None, 
                 energy_chart_width=None, energy_chart_height=None,
                 x_pygame_window=0, y_pygame_window=0):
        self.env = env
        self.cell_scale = cell_scale
        self.has_energy_chart = has_energy_chart
        self.file_name = 0

        # Initialize the Pygame window position
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x_pygame_window},{y_pygame_window}"

        # Grid size settings (defaults to environment size if not provided)
        self.width = grid_width if grid_width else env.x_grid_size * cell_scale
        self.height = grid_height if grid_height else env.y_grid_size * cell_scale
        
        # Ensure energy chart width and height have valid defaults
        self.width_energy_chart = energy_chart_width if energy_chart_width is not None else (2040 if has_energy_chart else 0)
        self.height_energy_chart = energy_chart_height if energy_chart_height is not None else self.height

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

        return np.transpose(new_observation, axes=(1, 0, 2)) if self.env.render_mode == "rgb_array" else None

    def _draw_grid(self):
        for x in range(self.env.x_grid_size):
            for y in range(self.env.y_grid_size):
                cell_rect = pygame.Rect(
                    self.cell_scale * x, self.cell_scale * y, self.cell_scale, self.cell_scale
                )
                pygame.draw.rect(self.screen, (255, 255, 255), cell_rect)
                pygame.draw.rect(self.screen, (192, 192, 192), cell_rect, 1)
        border_rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self.screen, (255, 0, 0), border_rect, 5)

    def _draw_observations(self, instances, color):
        for instance in instances:
            x, y = instance.position
            size = self.cell_scale * instance.observation_range
            patch = pygame.Surface((size, size))
            patch.set_alpha(128)
            patch.fill(color)
            offset = instance.observation_range / 2.0
            positions = [(x, y)]
            if self.env.is_torus:
                positions = self._get_torus_positions(x, y, instance.observation_range)
            for pos_x, pos_y in positions:
                self.screen.blit(
                    patch,
                    (self.cell_scale * (pos_x - offset + 0.5), self.cell_scale * (pos_y - offset + 0.5))
                )

    def _draw_instances(self, instances, color):
        for instance in instances:
            x, y = instance.position
            center = (
                int(self.cell_scale * x + self.cell_scale / 2),
                int(self.cell_scale * y + self.cell_scale / 2)
            )
            pygame.draw.circle(self.screen, color, center, int(self.cell_scale / 2.3))

    def _draw_energy_chart_predators(self, offset_x):
        x_pos, y_pos = self.width + offset_x, 0
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_bars(x_pos, y_pos, self.env.predator_type_nr, (255, 0, 0))

    def _draw_energy_chart_prey(self, offset_x):
        x_pos, y_pos = self.width + offset_x, 500
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_bars(x_pos, y_pos, self.env.prey_type_nr, (0, 0, 255))

    def _draw_energy_bars(self, x_pos, y_pos, agent_type, color):
        bar_width = 15
        spacing = 5
        max_energy = 100
        height = 400
        for i, name in enumerate(self.env.possible_agent_name_list_type[agent_type]):
            instance = self.env.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            pygame.draw.rect(self.screen, color, (x_pos + i * (bar_width + spacing), y_pos + height - bar_height, bar_width, bar_height))

    def _draw_agent_ids(self):
        font = pygame.font.SysFont("Comic Sans MS", self.cell_scale * 2 // 3)
        for agent_list in [
            self.env.active_agent_instance_list_type[self.env.predator_type_nr],
            self.env.active_agent_instance_list_type[self.env.prey_type_nr],
            self.env.active_agent_instance_list_type[self.env.grass_type_nr]
        ]:
            for instance in agent_list:
                x, y = instance.position
                pos_x, pos_y = self.cell_scale * x + self.cell_scale // 6, self.cell_scale * y + self.cell_scale // 1.2
                text = font.render(str(instance.agent_id_nr), False, (255, 255, 0))
                self.screen.blit(text, (pos_x, pos_y - self.cell_scale // 2))

    def _get_torus_positions(self, x, y, observation_range):
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
        return positions

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
