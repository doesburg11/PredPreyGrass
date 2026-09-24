import os
import pygame
from dataclasses import dataclass

ICON_DIR = os.path.join(os.path.dirname(__file__), *[os.pardir] * 4, "assets", "images", "icons")


@dataclass
class GuiStyle:
    margin_left: int = 10
    margin_top: int = 10
    margin_right: int = 340
    margin_bottom: int = 10
    legend_spacing: int = 30
    legend_font_size: int = 28
    legend_circle_radius: int = 10
    legend_square_size: int = 20
    tooltip_font_size: int = 28
    tooltip_padding: int = 4

    predator_male_color: tuple = (0, 90, 255)  # blue
    predator_female_color: tuple = (255, 105, 180)  # pink
    prey_color: tuple = (139, 90, 43)  # mammoth brown (chart line; keeps blue free for males)
    grass_color: tuple = (0, 128, 0)
    fruit_color: tuple = (255, 165, 0)  # orange
    grid_color: tuple = (200, 200, 200)
    background_color: tuple = (255, 255, 255)


class PyGameRenderer:
    def __init__(self, grid_size, cell_size=32, ennable_speed_slider=True):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.enable_speed_slider = ennable_speed_slider
        self.gui_style = GuiStyle()

        window_width = self.gui_style.margin_left + grid_size[0] * cell_size + self.gui_style.margin_right
        window_height = self.gui_style.margin_top + grid_size[1] * cell_size + self.gui_style.margin_bottom
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("PredPreyGrass Live Viewer — Predator Sexual Reproduction")

        self.font = pygame.font.SysFont(None, int(cell_size * 0.5))
        self.font_legend = pygame.font.SysFont(None, self.gui_style.legend_font_size)

        self.tooltip_font = pygame.font.SysFont(None, self.gui_style.tooltip_font_size)

        self.reference_energy_predator = 10.0
        self.reference_energy_prey = 3.0
        self.reference_energy_grass = 2.0
        self.reference_energy_fruit = 2.0


        self.previous_agent_energies = {}
        self.population_history_steps = []
        self.population_history_predator_male = []
        self.population_history_predator_female = []
        self.population_history_prey = []
        self.population_history_max_length = 1000

        self._icon_source = {
            "male": self._load_icon("male_symbol.png", self.gui_style.predator_male_color),
            "female": self._load_icon("female_symbol.png", self.gui_style.predator_female_color),
            "prey": self._load_icon("mammoth_prey.png"),
        }
        self._icon_cache = {}

        self.target_fps = 10
        self.slider_rect = None
        self.slider_max_fps = 60

    @staticmethod
    def _load_icon(filename, color=None):
        """Load an icon PNG; if `color` is given, recolor it, keeping its alpha shape."""
        icon = pygame.image.load(os.path.join(ICON_DIR, filename)).convert_alpha()
        if color is None:
            return icon
        return pygame.mask.from_surface(icon).to_surface(setcolor=color, unsetcolor=(0, 0, 0, 0))

    def _get_icon(self, sex, height):
        """Icon scaled (aspect preserved) to fit a box of `height` pixels."""
        key = (sex, height)
        if key not in self._icon_cache:
            src = self._icon_source[sex]
            scale = height / max(src.get_width(), src.get_height())
            size = (max(int(src.get_width() * scale), 1), max(int(src.get_height() * scale), 1))
            self._icon_cache[key] = pygame.transform.smoothscale(src, size)
        return self._icon_cache[key]

    def _blit_icon_centered(self, sex, height, x, y):
        icon = self._get_icon(sex, height)
        self.screen.blit(icon, icon.get_rect(center=(x, y)))

    def update(
        self,
        agent_positions,
        grass_positions,
        agent_energies=None,
        grass_energies=None,
        fruit_positions=None,
        fruit_energies=None,
        step=0,
        agents_just_ate=None,
    ):
        if agents_just_ate is None:
            agents_just_ate = set()
        if fruit_positions is None:
            fruit_positions = {}

        self.screen.fill(self.gui_style.background_color)

        num_predator_male = sum(1 for agent_id in agent_positions if "predator_male" in agent_id)
        num_predator_female = sum(1 for agent_id in agent_positions if "predator_female" in agent_id)
        num_prey = sum(1 for agent_id in agent_positions if "prey" in agent_id)

        self.population_history_steps.append(step)
        self.population_history_predator_male.append(num_predator_male)
        self.population_history_predator_female.append(num_predator_female)
        self.population_history_prey.append(num_prey)

        if len(self.population_history_steps) > self.population_history_max_length:
            self.population_history_steps.pop(0)
            self.population_history_predator_male.pop(0)
            self.population_history_predator_female.pop(0)
            self.population_history_prey.pop(0)

        self._draw_grid()
        self._draw_patches(grass_positions, grass_energies, self.gui_style.grass_color, self.reference_energy_grass)
        self._draw_patches(fruit_positions, fruit_energies, self.gui_style.fruit_color, self.reference_energy_fruit)
        self._draw_agents(agent_positions, agent_energies, agents_just_ate)
        self._draw_tooltip(agent_positions, grass_positions, fruit_positions, agent_energies, grass_energies, fruit_energies)
        self._draw_legend(step)

        pygame.display.set_caption(f"PredPreyGrass Live Viewer — Step {step}")
        pygame.display.flip()

    def _draw_grid(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(
                    self.gui_style.margin_left + x * self.cell_size,
                    self.gui_style.margin_top + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, self.gui_style.grid_color, rect, 1)

    def _draw_patches(self, positions, energies, color, reference_energy):
        for patch_id, pos in positions.items():
            x_pix = self.gui_style.margin_left + pos[0] * self.cell_size + self.cell_size // 2
            y_pix = self.gui_style.margin_top + pos[1] * self.cell_size + self.cell_size // 2
            energy = energies.get(patch_id, 0) if energies else 0
            size_factor = min(energy / reference_energy, 1.0)
            base_rect_size = self.cell_size * 0.8
            rect_size = base_rect_size * size_factor
            rect = pygame.Rect(x_pix - rect_size // 2, y_pix - rect_size // 2, rect_size, rect_size)
            pygame.draw.rect(self.screen, color, rect)

    def _draw_agents(self, agent_positions, agent_energies, agents_just_ate):
        for agent_id, pos in agent_positions.items():
            x_pix = self.gui_style.margin_left + pos[0] * self.cell_size + self.cell_size // 2
            y_pix = self.gui_style.margin_top + pos[1] * self.cell_size + self.cell_size // 2
            energy = agent_energies.get(agent_id, 0) if agent_energies else 0

            if "predator_male" in agent_id:
                color = self.gui_style.predator_male_color
                reference_energy = self.reference_energy_predator
            elif "predator_female" in agent_id:
                color = self.gui_style.predator_female_color
                reference_energy = self.reference_energy_predator
            elif "prey" in agent_id:
                color = self.gui_style.prey_color
                reference_energy = self.reference_energy_prey
            else:
                color = (0, 0, 0)
                reference_energy = 1.0

            size_factor = min(energy / reference_energy, 1.0)
            base_radius = self.cell_size // 2 - 2
            radius = int(base_radius * size_factor)

            if "predator_male" in agent_id:
                self._blit_icon_centered("male", max(2 * radius, 4), x_pix, y_pix)
            elif "predator_female" in agent_id:
                self._blit_icon_centered("female", max(2 * radius, 4), x_pix, y_pix)
            elif "prey" in agent_id:
                self._blit_icon_centered("prey", max(2 * radius, 4), x_pix, y_pix)
            else:
                pygame.draw.circle(self.screen, color, (x_pix, y_pix), max(radius, 2))

    def _draw_legend(self, step):
        x = self.gui_style.margin_left + self.grid_size[0] * self.cell_size + 20
        y = self.gui_style.margin_top + 10

        y = self._draw_legend_step_counter(x, y, step)
        y = self._draw_legend_agents(x, y)
        y = self._draw_legend_environment_elements(x, y)
        if self.enable_speed_slider:
            y = self._draw_legend_speed_slider(x, y)
        self._draw_legend_population_chart(x, y)

    def _draw_legend_step_counter(self, x, y, step):
        spacing = self.gui_style.legend_spacing
        font_large = self.font_legend
        step_label_surface = font_large.render("Step:", True, (0, 0, 0))
        self.screen.blit(step_label_surface, (x, y))

        label_width = step_label_surface.get_width()
        step_number_surface = font_large.render(f"{step}", True, (255, 0, 0))
        self.screen.blit(step_number_surface, (x + label_width + 5, y))

        return y + spacing

    def _draw_legend_agents(self, x, y):
        spacing = self.gui_style.legend_spacing
        r = self.gui_style.legend_circle_radius
        font = self.tooltip_font
        font_large = self.font_legend

        title_surface = font_large.render("Agent size depends on energy", True, (0, 0, 0))
        self.screen.blit(title_surface, (x, y))
        y += spacing

        self._blit_icon_centered("male", 2 * r, x + r, y + r)
        self.screen.blit(font.render("Predator (male, hunts + gathers)", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        self._blit_icon_centered("female", 2 * r, x + r, y + r)
        self.screen.blit(font.render("Predator (female, gathers only)", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        self._blit_icon_centered("prey", 2 * r, x + r, y + r)
        self.screen.blit(font.render("Prey (mammoth)", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        return y

    def _draw_legend_environment_elements(self, x, y):
        spacing = self.gui_style.legend_spacing
        r = self.gui_style.legend_circle_radius
        s = self.gui_style.legend_square_size
        font = self.tooltip_font

        pygame.draw.rect(self.screen, self.gui_style.grass_color, pygame.Rect(x + r - s // 2, y + r - s // 2, s, s))
        self.screen.blit(font.render("Grass (prey food)", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        pygame.draw.rect(self.screen, self.gui_style.fruit_color, pygame.Rect(x + r - s // 2, y + r - s // 2, s, s))
        self.screen.blit(font.render("Fruit (predator food)", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        return y

    def _draw_legend_speed_slider(self, x, y):
        spacing = self.gui_style.legend_spacing
        font = pygame.font.SysFont(None, 24)

        y += spacing

        slider_label_surface = font.render("Speed (steps/sec)", True, (0, 0, 0))
        self.screen.blit(slider_label_surface, (x, y))

        y += spacing

        slider_x = x
        slider_y = y
        slider_width = 200
        slider_height = 20

        pygame.draw.rect(self.screen, (180, 180, 180), pygame.Rect(slider_x, slider_y, slider_width, slider_height))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(slider_x, slider_y, slider_width, slider_height), 1)

        slider_max_fps = self.slider_max_fps
        ratio = (self.target_fps - 1) / (slider_max_fps - 1)
        handle_x = slider_x + int(ratio * slider_width)
        handle_y = slider_y + slider_height // 2

        pygame.draw.circle(self.screen, (50, 50, 250), (handle_x, handle_y), 16)

        fps_surface = font.render(f"{self.target_fps} FPS", True, (0, 0, 0))
        self.screen.blit(fps_surface, (slider_x + slider_width + 15, slider_y - 8))

        self.slider_rect = pygame.Rect(slider_x, slider_y, slider_width, slider_height)

        return y + spacing

    def _draw_legend_population_chart(self, x, y):
        chart_width = 260
        chart_height = 100
        chart_x = x + 30
        chart_y = y + 40
        spacing = self.gui_style.legend_spacing

        font_small = pygame.font.SysFont(None, int(self.gui_style.tooltip_font_size * 0.8), bold=False)
        title_surface = font_small.render("Predator (M/F) and prey population", True, (0, 0, 0))
        title_x = chart_x + chart_width // 2 - title_surface.get_width() // 2
        title_y = chart_y - spacing // 2 - 10
        self.screen.blit(title_surface, (title_x, title_y))

        pygame.draw.rect(self.screen, (230, 230, 230), pygame.Rect(chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(chart_x, chart_y, chart_width, chart_height), 1)

        max_agents = max(
            max(self.population_history_predator_male, default=1),
            max(self.population_history_predator_female, default=1),
            max(self.population_history_prey, default=1),
            1,
        )
        num_ticks = 5
        label_x = chart_x - 5

        for i in range(num_ticks + 1):
            value = int(i / num_ticks * max_agents)
            y_pos = chart_y + chart_height - int(i / num_ticks * chart_height)
            label_surface = font_small.render(f"{value}", True, (0, 0, 0))
            label_width = label_surface.get_width()
            self.screen.blit(label_surface, (label_x - label_width, y_pos - label_surface.get_height() // 2))

        x_tick_labels = [0, 200, 400, 600, 800, 1000]
        num_xticks = len(x_tick_labels)
        for i, label_value in enumerate(x_tick_labels):
            x_pos = chart_x + int(i / (num_xticks - 1) * chart_width)
            y_pos = chart_y + chart_height + 5
            label_surface = font_small.render(f"{label_value}", True, (0, 0, 0))
            label_width = label_surface.get_width()
            self.screen.blit(label_surface, (x_pos - label_width // 2, y_pos))

        if self.population_history_steps:
            for i in range(1, len(self.population_history_steps)):
                x1 = chart_x + int((i - 1) / self.population_history_max_length * chart_width)
                x2 = chart_x + int(i / self.population_history_max_length * chart_width)

                y1m = chart_y + chart_height - int(self.population_history_predator_male[i - 1] / max_agents * chart_height)
                y2m = chart_y + chart_height - int(self.population_history_predator_male[i] / max_agents * chart_height)
                pygame.draw.line(self.screen, self.gui_style.predator_male_color, (x1, y1m), (x2, y2m), 2)

                y1f = chart_y + chart_height - int(self.population_history_predator_female[i - 1] / max_agents * chart_height)
                y2f = chart_y + chart_height - int(self.population_history_predator_female[i] / max_agents * chart_height)
                pygame.draw.line(self.screen, self.gui_style.predator_female_color, (x1, y1f), (x2, y2f), 2)

                y1p = chart_y + chart_height - int(self.population_history_prey[i - 1] / max_agents * chart_height)
                y2p = chart_y + chart_height - int(self.population_history_prey[i] / max_agents * chart_height)
                pygame.draw.line(self.screen, self.gui_style.prey_color, (x1, y1p), (x2, y2p), 2)

        return chart_y + chart_height + self.gui_style.legend_spacing

    def _draw_tooltip(self, agent_positions, grass_positions, fruit_positions, agent_energies, grass_energies, fruit_energies):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x = (mouse_x - self.gui_style.margin_left) // self.cell_size
        grid_y = (mouse_y - self.gui_style.margin_top) // self.cell_size
        hovered_entity = None
        hovered_energy = 0.0

        for agent_id, pos in agent_positions.items():
            if pos == (grid_x, grid_y):
                hovered_entity = agent_id
                hovered_energy = agent_energies.get(agent_id, 0) if agent_energies else 0
                break

        if not hovered_entity:
            for grass_id, pos in grass_positions.items():
                if pos == (grid_x, grid_y):
                    hovered_entity = grass_id
                    hovered_energy = grass_energies.get(grass_id, 0) if grass_energies else 0
                    break

        if not hovered_entity and fruit_positions:
            for fruit_id, pos in fruit_positions.items():
                if pos == (grid_x, grid_y):
                    hovered_entity = fruit_id
                    hovered_energy = fruit_energies.get(fruit_id, 0) if fruit_energies else 0
                    break

        if hovered_entity:
            tooltip_line1 = self.tooltip_font.render(f"{hovered_entity}", True, (0, 0, 0))
            tooltip_line2 = self.tooltip_font.render(f"Energy: {hovered_energy:.2f}", True, (0, 0, 0))
            padding = self.gui_style.tooltip_padding
            width = max(tooltip_line1.get_width(), tooltip_line2.get_width())
            height = tooltip_line1.get_height() + tooltip_line2.get_height()
            tooltip_x = mouse_x + 10
            tooltip_y = mouse_y + 10
            bg_rect = pygame.Rect(tooltip_x - padding, tooltip_y - padding, width + 2 * padding, height + 2 * padding)
            pygame.draw.rect(self.screen, (255, 255, 200), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)
            self.screen.blit(tooltip_line1, (tooltip_x, tooltip_y))
            self.screen.blit(tooltip_line2, (tooltip_x, tooltip_y + tooltip_line1.get_height()))

    def close(self):
        pygame.quit()


class ViewerControlHelper:
    """
    Helper class to manage viewer control:
    - Pause / Play (SPACE)
    - Single Step (RIGHT arrow)
    - Step Backward (LEFT arrow)
    - Quit (window close or ESC)
    """

    def __init__(self, initial_paused=False):
        self.paused = initial_paused
        self.step_once = False
        self.step_backward = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("[ViewerControl] Quit detected — exiting.")
                pygame.quit()
                exit(0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("[ViewerControl] ESC pressed — exiting.")
                    pygame.quit()
                    exit(0)

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"[ViewerControl] Pause {'ON' if self.paused else 'OFF'}")

                elif event.key == pygame.K_RIGHT:
                    self.paused = True
                    self.step_once = True
                    print("[ViewerControl] Single Step")

                elif event.key == pygame.K_LEFT:
                    self.paused = True
                    self.step_backward = True
                    print("[ViewerControl] Step Backward")


class LoopControlHelper:
    """
    Small helper to manage 'simulation terminated' state safely, and provide
    a uniform should_step() pattern across scripts. See
    base_environment_step_energy/utils/pygame_grid_renderer_rllib.py for the
    full rationale (unchanged here).
    """

    def __init__(self):
        self.simulation_terminated = False

    def update_simulation_terminated(self, terminations, truncations):
        self.simulation_terminated = terminations.get("__all__", False) or truncations.get("__all__", False)

    def should_step(self, control):
        return not control.paused or control.step_once
