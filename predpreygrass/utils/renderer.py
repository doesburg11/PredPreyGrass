import pygame
import numpy as np
import os

class Renderer:
    def __init__(self, env, cell_scale=40, show_energy_chart=True, x_pygame_window=0, y_pygame_window=0):
        self.env = env
        self.cell_scale = cell_scale
        self.show_energy_chart = show_energy_chart
        self.file_name = 0

        # Initialize the Pygame window position
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (x_pygame_window, y_pygame_window)

        # Pygame screen settings
        self.width = env.x_grid_size * self.cell_scale
        self.height = env.y_grid_size * self.cell_scale
        self.width_energy_chart = 1800 if show_energy_chart else 0
        self.height_energy_chart: int = self.cell_scale * self.env.y_grid_size

        self.save_image_steps = False 

        pygame.init()
        if env.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width + self.width_energy_chart, self.height))
            pygame.display.set_caption("PredPreyGrass - create agents")
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

        if self.show_energy_chart:
            self._draw_energy_chart(0)

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
            self.screen.blit(
                patch,
                (self.cell_scale * (x - offset + 0.5), self.cell_scale * (y - offset + 0.5))
            )

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

    def _draw_energy_labels(self, x_pos, y_pos):
        predator_label_font = pygame.font.Font(None, 35)
        prey_label_font = pygame.font.Font(None, 35)
        predator_label = predator_label_font.render("Predators", True, (255, 0, 0))
        prey_label = prey_label_font.render("Prey", True, (0, 0, 255))
        self.screen.blit(predator_label, (x_pos, y_pos))
        self.screen.blit(prey_label, (x_pos + (len(self.env.possible_agent_name_list_type[self.env.predator_type_nr]) * 40), y_pos))

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

    def _draw_energy_chart(self, offset_x):
        x_pos, y_pos = self.cell_scale * self.env.x_grid_size + offset_x, 0
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        self._draw_energy_title(x_pos + 750, 20)
        self._draw_energy_bars(x_pos + 100, 50, self.width_energy_chart - 100, 500)
        self._draw_energy_labels(x_pos + 100, 585)

    def _draw_energy_title(self, x_pos, y_pos):
        title_font = pygame.font.Font(None, 40)
        title = title_font.render("Energy Level Agents", True, (0, 0, 0))
        self.screen.blit(title, (x_pos, y_pos))

    def _draw_energy_labels(self, x_pos, y_pos):
        predator_label_font = pygame.font.Font(None, 35)
        prey_label_font = pygame.font.Font(None, 35)
        predator_label = predator_label_font.render("Predators", True, (255, 0, 0))
        prey_label = prey_label_font.render("Prey", True, (0, 0, 255))
        self.screen.blit(predator_label, (x_pos, y_pos))
        self.screen.blit(prey_label, (x_pos + (len(self.env.possible_agent_name_list_type[self.env.predator_type_nr]) * 40), y_pos))

    def _draw_energy_bars(self, x_pos, y_pos, width, height):
        bar_width, offset, max_energy, red, blue = 20, 20, 30, (255, 0, 0), (0, 0, 255)
        y_axis_x, x_axis_y = x_pos - 20, y_pos + height

        # Draw Axes
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, y_pos, 5, height))
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, x_axis_y, width + 40, 5))

        # Draw Bars and Labels for Predators
        for i, name in enumerate(self.env.possible_agent_name_list_type[self.env.predator_type_nr]):
            instance = self.env.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            bar_x, bar_y = x_pos + i * (bar_width + offset), y_pos + height - bar_height
            pygame.draw.rect(self.screen, red, (bar_x, bar_y, bar_width, bar_height))
            label_x, label_y = bar_x, x_axis_y + 10
            label = pygame.font.Font(None, 30).render(str(instance.agent_id_nr), True, red)
            self.screen.blit(label, (label_x, label_y))

        # Draw Bars and Labels for Prey
        for i, name in enumerate(self.env.possible_agent_name_list_type[self.env.prey_type_nr]):
            instance = self.env.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            bar_x, bar_y = x_pos + (i + len(self.env.possible_agent_name_list_type[self.env.predator_type_nr])) * (bar_width + offset), y_pos + height - bar_height
            pygame.draw.rect(self.screen, blue, (bar_x, bar_y, bar_width, bar_height))
            label_x, label_y = bar_x, x_axis_y + 10
            label = pygame.font.Font(None, 30).render(str(instance.agent_id_nr), True, blue)
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
        print(f"{self.file_name}.png saved")
        pygame.image.save(self.screen, f"./assets/images/{self.file_name}.png")

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

