import pygame
from dataclasses import dataclass


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

    predator_color: tuple = (255, 0, 0)
    prey_color: tuple = (0, 0, 255)
    grass_color: tuple = (0, 128, 0)
    predator_speed_1_color: tuple = (255, 0, 0)  # Bright red
    predator_speed_2_color: tuple = (165, 0, 0)
    prey_speed_1_color: tuple = (0, 0, 255)  # Bright blue
    prey_speed_2_color: tuple = (100, 100, 255)
    grid_color: tuple = (200, 200, 200)
    background_color: tuple = (255, 255, 255)
    halo_reproduction_color: tuple = (255, 0, 0)  # Gold
    halo_eating_color: tuple = (0, 128, 0)  # Bright green
    halo_reproduction_thickness: int = 3
    halo_eating_thickness: int = 3


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
        pygame.display.set_caption("PredPreyGrass Live Viewer")

        self.font = pygame.font.SysFont(None, int(cell_size * 0.5))
        self.font_legend = pygame.font.SysFont(None, self.gui_style.legend_font_size)

        self.tooltip_font = pygame.font.SysFont(None, self.gui_style.tooltip_font_size)

        self.reference_energy_predator = 10.0  # referenc wrt size of predators
        self.reference_energy_prey = 3.0
        self.reference_energy_grass = 2.0

        self.halo_trigger_fraction = 0.9
        self.predator_creation_energy_threshold = 12.0
        self.prey_creation_energy_threshold = 8.0

        self.previous_agent_energies = {}
        self.population_history_steps = []
        self.population_history_predators = []
        self.population_history_prey = []
        self.population_history_max_length = 1000
        self.population_pct_speed2_predators = []
        self.population_pct_speed2_prey = []

        self.target_fps = 10  # Default FPS
        self.slider_rect = None  # Will be defined in _draw_legend()
        self.slider_max_fps = 60

    def _using_speed_prefix(self, agent_positions):
        """Return True if any agent_id contains speed info."""
        return any("speed_1" in aid or "speed_2" in aid for aid in agent_positions.keys())

    def update(self, agent_positions, grass_positions, agent_energies=None, grass_energies=None, step=0, agents_just_ate=None):
        if agents_just_ate is None:
            agents_just_ate = set()

        self.screen.fill(self.gui_style.background_color)

        # Count predator and prey population
        num_predators = sum(1 for agent_id in agent_positions if "predator" in agent_id)
        num_prey = sum(1 for agent_id in agent_positions if "prey" in agent_id)
        num_speed2_predators = sum(1 for aid in agent_positions if "predator" in aid and "speed_2" in aid)
        num_speed2_prey = sum(1 for aid in agent_positions if "prey" in aid and "speed_2" in aid)

        pct_predators_speed2 = (num_speed2_predators / num_predators) if num_predators > 0 else 0
        pct_prey_speed2 = (num_speed2_prey / num_prey) if num_prey > 0 else 0

        # Update population history
        self.population_pct_speed2_predators.append(pct_predators_speed2)
        self.population_pct_speed2_prey.append(pct_prey_speed2)

        # Keep history bounded
        if len(self.population_pct_speed2_predators) > self.population_history_max_length:
            self.population_pct_speed2_predators.pop(0)
            self.population_pct_speed2_prey.pop(0)

        self.population_history_steps.append(step)
        self.population_history_predators.append(num_predators)
        self.population_history_prey.append(num_prey)

        # Keep history length bounded
        if len(self.population_history_steps) > self.population_history_max_length:
            self.population_history_steps.pop(0)
            self.population_history_predators.pop(0)
            self.population_history_prey.pop(0)

        self._draw_grid()
        self._draw_grass(grass_positions, grass_energies)
        self._draw_agents(agent_positions, agent_energies, agents_just_ate)
        self._draw_tooltip(agent_positions, grass_positions, agent_energies, grass_energies)
        self._draw_legend(step, agent_positions)

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

    def _draw_grass(self, grass_positions, grass_energies):
        for grass_id, pos in grass_positions.items():
            x_pix = self.gui_style.margin_left + pos[0] * self.cell_size + self.cell_size // 2
            y_pix = self.gui_style.margin_top + pos[1] * self.cell_size + self.cell_size // 2
            color = self.gui_style.grass_color
            energy = grass_energies.get(grass_id, 0) if grass_energies else 0
            size_factor = min(energy / self.reference_energy_grass, 1.0)
            base_rect_size = self.cell_size * 0.8
            rect_size = base_rect_size * size_factor
            rect = pygame.Rect(x_pix - rect_size // 2, y_pix - rect_size // 2, rect_size, rect_size)
            pygame.draw.rect(self.screen, color, rect)

    def _draw_agents(self, agent_positions, agent_energies, agents_just_ate):
        for agent_id, pos in agent_positions.items():
            x_pix = self.gui_style.margin_left + pos[0] * self.cell_size + self.cell_size // 2
            y_pix = self.gui_style.margin_top + pos[1] * self.cell_size + self.cell_size // 2
            energy = agent_energies.get(agent_id, 0) if agent_energies else 0

            # Determine type and speed
            if "predator" in agent_id:
                color = self.gui_style.predator_color  # Default predator color
                if "speed_1" in agent_id:
                    color = self.gui_style.predator_speed_1_color
                elif "speed_2" in agent_id:
                    color = self.gui_style.predator_speed_2_color
                reference_energy = self.reference_energy_predator
                threshold = self.predator_creation_energy_threshold

            elif "prey" in agent_id:
                color = self.gui_style.prey_color  # Default prey color
                if "speed_1" in agent_id:
                    color = self.gui_style.prey_speed_1_color
                elif "speed_2" in agent_id:
                    color = self.gui_style.prey_speed_2_color
                reference_energy = self.reference_energy_prey
                threshold = self.prey_creation_energy_threshold

            size_factor = min(energy / reference_energy, 1.0)
            base_radius = self.cell_size // 2 - 2
            radius = int(base_radius * size_factor)

            # Draw main body
            pygame.draw.circle(self.screen, color, (x_pix, y_pix), max(radius, 2))

            # Eating halo (green ring)
            if agent_id in agents_just_ate:
                pygame.draw.circle(
                    self.screen,
                    self.gui_style.halo_eating_color,
                    (x_pix, y_pix),
                    max(radius + 5, 6),
                    width=self.gui_style.halo_eating_thickness,
                )

            # Reproduction halo (red ring)
            if threshold and energy >= threshold * self.halo_trigger_fraction:
                pygame.draw.circle(
                    self.screen,
                    self.gui_style.halo_reproduction_color,
                    (x_pix, y_pix),
                    max(radius + 5, 6),
                    width=self.gui_style.halo_reproduction_thickness,
                )

    def _draw_legend(self, step, agent_positions):
        x = self.gui_style.margin_left + self.grid_size[0] * self.cell_size + 20
        y = self.gui_style.margin_top + 10

        y = self._draw_legend_step_counter(x, y, step)

        y = self._draw_legend_agents(x, y, agent_positions)

        y = self._draw_legend_environment_elements(x, y)

        if self.enable_speed_slider:
            y = self._draw_legend_speed_slider(x, y)

        y = self._draw_legend_population_chart(x, y)

        if self._using_speed_prefix(agent_positions):
            y = self._draw_legend_speed2_percent_chart(x, y)

    def _draw_legend_step_counter(self, x, y, step):
        spacing = self.gui_style.legend_spacing
        font_large = self.font_legend
        step_label_surface = font_large.render("Step:", True, (0, 0, 0))
        self.screen.blit(step_label_surface, (x, y))

        label_width = step_label_surface.get_width()
        step_number_surface = font_large.render(f"{step}", True, (255, 0, 0))
        self.screen.blit(step_number_surface, (x + label_width + 5, y))  # +5 pixels spacing

        return y + spacing

    def _draw_legend_agents(self, x, y, agent_positions):
        spacing = self.gui_style.legend_spacing
        r = self.gui_style.legend_circle_radius
        font = self.tooltip_font
        font_large = self.font_legend

        # Draw legend title
        title_surface = font_large.render("Agent size depends on energy", True, (0, 0, 0))
        self.screen.blit(title_surface, (x, y))
        y += spacing

        is_speed_based = self._using_speed_prefix(agent_positions)

        if is_speed_based:
            # Speed-specific predator colors
            pygame.draw.circle(self.screen, self.gui_style.predator_speed_1_color, (x + r, y + r), r)
            self.screen.blit(font.render("Predator (Speed 1)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            pygame.draw.circle(self.screen, self.gui_style.predator_speed_2_color, (x + r, y + r), r)
            self.screen.blit(font.render("Predator (Speed 2)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            # Speed-specific prey colors
            pygame.draw.circle(self.screen, self.gui_style.prey_speed_1_color, (x + r, y + r), r)
            self.screen.blit(font.render("Prey (Speed 1)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            pygame.draw.circle(self.screen, self.gui_style.prey_speed_2_color, (x + r, y + r), r)
            self.screen.blit(font.render("Prey (Speed 2)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

        else:
            # Compact predator/prey legend
            pygame.draw.circle(self.screen, self.gui_style.predator_color, (x + r, y + r), r)
            self.screen.blit(font.render("Predator", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            pygame.draw.circle(self.screen, self.gui_style.prey_color, (x + r, y + r), r)
            self.screen.blit(font.render("Prey", True, (0, 0, 0)), (x + 30, y))
            y += spacing

        # Reproduction halo
        pygame.draw.circle(
            self.screen,
            self.gui_style.halo_reproduction_color,
            (x + r, y + r),
            r + 2,
            width=self.gui_style.halo_reproduction_thickness,
        )
        self.screen.blit(font.render("Close to reproduction halo", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        # Eating halo
        pygame.draw.circle(
            self.screen, self.gui_style.halo_eating_color, (x + r, y + r), r + 2, width=self.gui_style.halo_eating_thickness
        )
        self.screen.blit(font.render("Eating halo", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        return y

    def _draw_legend_environment_elements(self, x, y):
        spacing = self.gui_style.legend_spacing
        r = self.gui_style.legend_circle_radius
        s = self.gui_style.legend_square_size
        font = self.tooltip_font

        # Grass
        pygame.draw.rect(self.screen, self.gui_style.grass_color, pygame.Rect(x + r - s // 2, y + r - s // 2, s, s))
        self.screen.blit(font.render("Grass", True, (0, 0, 0)), (x + 30, y))
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

        return y + spacing  # Return new Y position for chart continuation

    def _draw_legend_population_chart(self, x, y):
        chart_width = 260
        chart_height = 100
        chart_x = x + 30
        chart_y = y + 40
        spacing = self.gui_style.legend_spacing

        # Chart title
        font_small = pygame.font.SysFont(None, int(self.gui_style.tooltip_font_size * 0.8), bold=False)
        title_surface = font_small.render("Total predator and prey population", True, (0, 0, 0))
        title_x = chart_x + chart_width // 2 - title_surface.get_width() // 2
        title_y = chart_y - spacing // 2 - 10
        self.screen.blit(title_surface, (title_x, title_y))

        # Chart box
        pygame.draw.rect(self.screen, (230, 230, 230), pygame.Rect(chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(chart_x, chart_y, chart_width, chart_height), 1)

        # Y axis
        max_agents = max(max(self.population_history_predators, default=1), max(self.population_history_prey, default=1), 1)
        num_ticks = 5
        label_x = chart_x - 5
        font_small = pygame.font.SysFont(None, int(self.gui_style.tooltip_font_size * 0.8), bold=False)

        for i in range(num_ticks + 1):
            value = int(i / num_ticks * max_agents)
            y_pos = chart_y + chart_height - int(i / num_ticks * chart_height)
            label_surface = font_small.render(f"{value}", True, (0, 0, 0))
            label_width = label_surface.get_width()
            self.screen.blit(label_surface, (label_x - label_width, y_pos - label_surface.get_height() // 2))

        # X ticks
        x_tick_labels = [0, 200, 400, 600, 800, 1000]
        num_xticks = len(x_tick_labels)
        for i, label_value in enumerate(x_tick_labels):
            x_pos = chart_x + int(i / (num_xticks - 1) * chart_width)
            y_pos = chart_y + chart_height + 5
            label_surface = font_small.render(f"{label_value}", True, (0, 0, 0))
            label_width = label_surface.get_width()
            self.screen.blit(label_surface, (x_pos - label_width // 2, y_pos))

        # Draw line chart
        if self.population_history_steps:
            for i in range(1, len(self.population_history_steps)):
                x1 = chart_x + int((i - 1) / self.population_history_max_length * chart_width)
                x2 = chart_x + int(i / self.population_history_max_length * chart_width)
                y1p = chart_y + chart_height - int(self.population_history_predators[i - 1] / max_agents * chart_height)
                y2p = chart_y + chart_height - int(self.population_history_predators[i] / max_agents * chart_height)
                pygame.draw.line(self.screen, self.gui_style.predator_speed_1_color, (x1, y1p), (x2, y2p), 2)
                y1pr = chart_y + chart_height - int(self.population_history_prey[i - 1] / max_agents * chart_height)
                y2pr = chart_y + chart_height - int(self.population_history_prey[i] / max_agents * chart_height)
                pygame.draw.line(self.screen, self.gui_style.prey_speed_1_color, (x1, y1pr), (x2, y2pr), 2)

        return chart_y + chart_height + self.gui_style.legend_spacing

    def _draw_legend_speed2_percent_chart(self, x, y):
        chart_width = 260
        chart_height = 100
        chart_x = x + 30
        chart_y = y + 40
        spacing = self.gui_style.legend_spacing

        font_small = pygame.font.SysFont(None, int(self.gui_style.tooltip_font_size * 0.8), bold=False)
        title_surface = font_small.render("Speed 2 predator and prey population (%)", True, (0, 0, 0))
        title_x = chart_x + chart_width // 2 - title_surface.get_width() // 2
        title_y = chart_y - spacing // 2 - 10
        self.screen.blit(title_surface, (title_x, title_y))

        pygame.draw.rect(self.screen, (230, 230, 230), pygame.Rect(chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(chart_x, chart_y, chart_width, chart_height), 1)

        # Y ticks (0% to 100%)
        for i in range(0, 6):
            pct = i * 0.2
            y_pos = chart_y + chart_height - int(pct * chart_height)
            label_surface = font_small.render(f"{int(pct * 100)}%", True, (0, 0, 0))
            label_x = chart_x - 6 - label_surface.get_width()  # right-align
            self.screen.blit(label_surface, (label_x, y_pos - 6))

        # X ticks
        x_tick_labels = [0, 200, 400, 600, 800, 1000]
        num_xticks = len(x_tick_labels)
        for i, label_value in enumerate(x_tick_labels):
            x_pos = chart_x + int(i / (num_xticks - 1) * chart_width)
            y_pos = chart_y + chart_height + 5
            label_surface = font_small.render(f"{label_value}", True, (0, 0, 0))
            self.screen.blit(label_surface, (x_pos - label_surface.get_width() // 2, y_pos))

        # Draw lines
        for i in range(1, len(self.population_pct_speed2_predators)):
            x1 = chart_x + int((i - 1) / self.population_history_max_length * chart_width)
            x2 = chart_x + int(i / self.population_history_max_length * chart_width)

            y1_pred = chart_y + chart_height - int(self.population_pct_speed2_predators[i - 1] * chart_height)
            y2_pred = chart_y + chart_height - int(self.population_pct_speed2_predators[i] * chart_height)
            pygame.draw.line(self.screen, self.gui_style.predator_speed_2_color, (x1, y1_pred), (x2, y2_pred), 2)

            y1_prey = chart_y + chart_height - int(self.population_pct_speed2_prey[i - 1] * chart_height)
            y2_prey = chart_y + chart_height - int(self.population_pct_speed2_prey[i] * chart_height)
            pygame.draw.line(self.screen, self.gui_style.prey_speed_2_color, (x1, y1_prey), (x2, y2_prey), 2)

        return chart_y + chart_height + spacing

    def _draw_tooltip(self, agent_positions, grass_positions, agent_energies, grass_energies):
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
    - Quit (window close or ESC)

    Usage:

    control = ViewerControlHelper()

    for step in ...:
        control.handle_events()

        if not control.paused or control.step_once:
            # env.step()
            # viewer.update()
            control.step_once = False
        else:
            # viewer.update()
            pygame.time.wait(50)
    """

    def __init__(self, initial_paused=False):
        self.paused = initial_paused
        self.step_once = False
        self.step_backward = False
        self.fps_slider_rect = None
        self.fps_slider_update_fn = None
        self.is_dragging_slider = False

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
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.fps_slider_rect and self.fps_slider_rect.collidepoint(event.pos):
                        self.is_dragging_slider = True
                        self._update_fps_slider(event.pos[0])

            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging_slider:
                    self._update_fps_slider(event.pos[0])

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.is_dragging_slider:
                    self.is_dragging_slider = False

    def _update_fps_slider(self, slider_x):
        slider_max_fps = self.visualizer.slider_max_fps
        slider_start_x = self.fps_slider_rect.left
        slider_width = self.fps_slider_rect.width
        ratio = (slider_x - slider_start_x) / slider_width
        ratio = min(max(ratio, 0.0), 1.0)
        new_fps = int(1 + ratio * (slider_max_fps - 1))
        self.fps_slider_update_fn(new_fps)
        print(f"[ViewerControl] Target FPS set to {new_fps}")


class LoopControlHelper:
    """
    LoopControlHelper

    Purpose:
    - Provide a small helper to manage 'simulation terminated' state safely.
    - Avoid accidental termination of the main loop when paused.
    - Provide 'should_step()' logic for consistent handling of pause/play/step.

    Why:
    - Without this helper, many users accidentally update 'done' on every loop iteration.
      → Result: when you pause, step, or hover, your loop can exit because 'done' was True from a previous step.
    - The correct pattern is to only update 'simulation_terminated' **after env.step()**, not in every loop.
    - This helper encapsulates that safe pattern.
    - It also gives a reusable 'should_step()' function → makes the loop structure uniform across scripts.

    Typical usage:

    control = ViewerControlHelper()
    loop_helper = LoopControlHelper()

    while not loop_helper.simulation_terminated:
        control.handle_events()

        if loop_helper.should_step(control):
            obs, rewards, terminations, truncations, info = env.step(action_dict)

            live_viewer.update(...)
            loop_helper.update_simulation_terminated(terminations, truncations)

            control.step_once = False
            ...

        else:
            live_viewer.update(...)
            pygame.time.wait(50)

    Summary:
    - This helper is very small, but ensures your viewer loops behave safely.
    - Prevents bugs when combining pause/play/step with 'done' checking.
    - You can use it in all viewer-based scripts (random_policy, evaluate_ppo_from_checkpoint, etc.).
    """

    def __init__(self):
        self.simulation_terminated = False

    def update_simulation_terminated(self, terminations, truncations):
        """Update termination flag — should be called only after env.step()."""
        self.simulation_terminated = terminations.get("__all__", False) or truncations.get("__all__", False)

    def should_step(self, control):
        """
        Determine if env.step() should be performed in this loop iteration.
        Returns True if:
        - not paused
        - or single step requested.
        """
        return not control.paused or control.step_once
