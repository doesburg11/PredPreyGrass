import pygame
import numpy as np

def render(self):
    def _draw_energy_title(x_pos, y_pos):
        title_font = pygame.font.Font(None, 40)
        title = title_font.render("Energy Level Agents", True, (0, 0, 0))
        self.screen.blit(title, (x_pos, y_pos))

    def _draw_energy_labels(x_pos, y_pos):
        predator_label_font = pygame.font.Font(None, 35)
        prey_label_font = pygame.font.Font(None, 35)
        predator_label = predator_label_font.render("Predators", True, (255, 0, 0))
        prey_label = prey_label_font.render("Prey", True, (0, 0, 255))
        self.screen.blit(predator_label, (x_pos, y_pos))
        self.screen.blit(prey_label, (x_pos + (len(self.possible_predator_name_list) * 40), y_pos))


    def _initialize_screen():
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.cell_scale * self.x_grid_size + self.width_energy_chart, self.cell_scale * self.y_grid_size)
            )
            pygame.display.set_caption("PredPreyGrass - create agents")
        else:
            self.screen = pygame.Surface(
                (self.cell_scale * self.x_grid_size, self.cell_scale * self.y_grid_size)
            )

    def _draw_grid():
        for x in range(self.x_grid_size):
            for y in range(self.y_grid_size):
                cell_rect = pygame.Rect(
                    self.cell_scale * x, self.cell_scale * y, self.cell_scale, self.cell_scale
                )
                pygame.draw.rect(self.screen, (255, 255, 255), cell_rect)
                pygame.draw.rect(self.screen, (192, 192, 192), cell_rect, 1)
        border_rect = pygame.Rect(0, 0, self.cell_scale * self.x_grid_size, self.cell_scale * self.y_grid_size)
        pygame.draw.rect(self.screen, (255, 0, 0), border_rect, 5)

    def _draw_observations(instances, color):
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

    def _draw_instances(instances, color):
        for instance in instances:
            x, y = instance.position
            center = (
                int(self.cell_scale * x + self.cell_scale / 2),
                int(self.cell_scale * y + self.cell_scale / 2)
            )
            pygame.draw.circle(self.screen, color, center, int(self.cell_scale / 2.3))

    def _draw_agent_ids():
        font = pygame.font.SysFont("Comic Sans MS", self.cell_scale * 2 // 3)
        for agent_list in [self.active_predator_instance_list, self.active_prey_instance_list, self.active_grass_instance_list]:
            for instance in agent_list:
                x, y = instance.position
                pos_x, pos_y = self.cell_scale * x + self.cell_scale // 6, self.cell_scale * y + self.cell_scale // 1.2
                text = font.render(str(instance.agent_id_nr), False, (255, 255, 0))
                self.screen.blit(text, (pos_x, pos_y - self.cell_scale // 2))

    def _draw_energy_chart(offset_x):
        x_pos, y_pos = self.cell_scale * self.x_grid_size + offset_x, 0
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x_pos, y_pos, self.width_energy_chart, self.height_energy_chart))
        _draw_energy_title(x_pos + 750, 20)
        _draw_energy_bars(x_pos + 100, 50, self.width_energy_chart - 100, 500)
        _draw_energy_labels(x_pos + 100, 585)

    def _draw_energy_bars(x_pos, y_pos, width, height):
        bar_width, offset, max_energy, red, blue = 20, 20, 30, (255, 0, 0), (0, 0, 255)
        y_axis_x, x_axis_y = x_pos - 20, y_pos + height

        # Draw Axes
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, y_pos, 5, height))
        pygame.draw.rect(self.screen, (0, 0, 0), (y_axis_x, x_axis_y, width + 40, 5))

        # Draw Bars and Labels for Predators
        for i, name in enumerate(self.possible_predator_name_list):
            instance = self.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            bar_x, bar_y = x_pos + i * (bar_width + offset), y_pos + height - bar_height
            pygame.draw.rect(self.screen, red, (bar_x, bar_y, bar_width, bar_height))
            label_x, label_y = bar_x, x_axis_y + 10
            label = pygame.font.Font(None, 30).render(str(instance.agent_id_nr), True, red)
            self.screen.blit(label, (label_x, label_y))

        # Draw Bars and Labels for Prey
        for i, name in enumerate(self.possible_prey_name_list):
            instance = self.agent_name_to_instance_dict[name]
            bar_height = (instance.energy / max_energy) * height
            bar_x, bar_y = x_pos + (i + len(self.possible_predator_name_list)) * (bar_width + offset), y_pos + height - bar_height
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

    def _save_image():
        self.file_name += 1
        print(f"{self.file_name}.png saved")
        pygame.image.save(self.screen, f"./assets/images/{self.file_name}.png")

    if self.render_mode is None:
        print("Render mode is not specified.")
        return

    if self.screen is None:
        _initialize_screen()

    _draw_grid()
    _draw_observations(self.active_prey_instance_list, (72, 152, 255))
    _draw_observations(self.active_predator_instance_list, (255, 152, 72))
    _draw_instances(self.active_grass_instance_list, (0, 128, 0))
    _draw_instances(self.active_prey_instance_list, (0, 0, 255))
    _draw_instances(self.active_predator_instance_list, (255, 0, 0))
    _draw_agent_ids()
    if self.show_energy_chart:
        _draw_energy_chart(0)

    observation = pygame.surfarray.pixels3d(self.screen)
    new_observation = np.copy(observation)
    del observation
    if self.render_mode == "human":
        pygame.event.pump()
        pygame.display.update()
        if self.save_image_steps:
            _save_image()

    return np.transpose(new_observation, axes=(1, 0, 2)) if self.render_mode == "rgb_array" else None
