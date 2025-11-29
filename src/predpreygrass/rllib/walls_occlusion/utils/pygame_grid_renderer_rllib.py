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
    predator_type_1_color: tuple = (255, 0, 0)  # Bright red
    predator_type_2_color: tuple = (165, 0, 0)
    prey_type_1_color: tuple = (0, 0, 255)  # Bright blue
    prey_type_2_color: tuple = (100, 100, 255)
    grid_color: tuple = (200, 200, 200)
    background_color: tuple = (255, 255, 255)
    halo_reproduction_color: tuple = (255, 0, 0)
    halo_eating_color: tuple = (0, 128, 0)  # Bright green
    halo_reproduction_thickness: int = 3
    halo_eating_thickness: int = 3
    wall_color: tuple = (80, 80, 80)  # Dark gray for walls


class PyGameRenderer:
    def __init__(
        self,
        grid_size,
        cell_size=32,
        enable_speed_slider=True,
        enable_tooltips=True,
        max_steps=None,
        predator_obs_range=None,
        prey_obs_range=None,
        show_fov=True,
        fov_alpha=70,
        fov_agents=None,
        fov_respect_walls=False,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.enable_speed_slider = enable_speed_slider
        self.enable_tooltips = enable_tooltips
        self.gui_style = GuiStyle()

        # Observation/FOV overlay configuration
        self.predator_obs_range = predator_obs_range
        self.prey_obs_range = prey_obs_range
        self.show_fov = show_fov
        self.fov_alpha = fov_alpha  # 0-255
        # Optional: restrict FOV overlay to specific agent IDs (exact match). If None, draw for all.
        self.fov_agents = set(fov_agents) if fov_agents is not None else None
        # If True, clip FOV overlay by line-of-sight (walls block further cells)
        self.fov_respect_walls = fov_respect_walls
        # Per-species FOV colors (RGBA)
        self.fov_color_predator = (255, 0, 0, self.fov_alpha)   # Red with alpha
        self.fov_color_prey = (0, 0, 255, self.fov_alpha)       # Blue with alpha

        # Cached walls (set of (x,y)) updated every frame in update(); used only when fov_respect_walls=True
        self._walls_set = set()

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
        self.population_pct_type2_predators = []
        self.population_pct_type2_prey = []

        self.target_fps = 10  # Default FPS
        self.slider_rect = None  # Will be defined in _draw_legend()
        self.slider_max_fps = 60
        self.population_history_max_length = max_steps if max_steps else 1000

        self.grid_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(
                    self.gui_style.margin_left + x * self.cell_size,
                    self.gui_style.margin_top + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.grid_surface, self.gui_style.grid_color, rect, 1)

    def _using_type_prefix(self, step_data):
        return any("type_1" in aid or "type_2" in aid for aid in step_data.keys())

    def update(self, grass_positions, grass_energies=None, step=0, agents_just_ate=None, per_step_agent_data=None, walls=None):
        step_data = per_step_agent_data[step - 1]
        if agents_just_ate is None:
            agents_just_ate = set()

        # Cache walls for FOV LOS clipping (if any)
        if walls:
            if isinstance(walls, dict):
                self._walls_set = {tuple(map(int, w)) for w in walls.values()}
            else:
                self._walls_set = {tuple(map(int, w)) for w in walls}
        else:
            self._walls_set = set()

        self.screen.fill(self.gui_style.background_color)

        # Count predator and prey population using step_data
        num_predators = num_prey = num_type2_predators = num_type2_prey = 0
        for agent_id in step_data:
            if "predator" in agent_id:
                num_predators += 1
                if "type_2" in agent_id:
                    num_type2_predators += 1
            elif "prey" in agent_id:
                num_prey += 1
                if "type_2" in agent_id:
                    num_type2_prey += 1

        pct_predators_type2 = (num_type2_predators / num_predators) if num_predators > 0 else 0
        pct_prey_type2 = (num_type2_prey / num_prey) if num_prey > 0 else 0

        # Update population history
        self.population_pct_type2_predators.append(pct_predators_type2)
        self.population_pct_type2_prey.append(pct_prey_type2)

        # Keep history bounded
        if len(self.population_pct_type2_predators) > self.population_history_max_length:
            self.population_pct_type2_predators.pop(0)
            self.population_pct_type2_prey.pop(0)

        self.population_history_steps.append(step)
        self.population_history_predators.append(num_predators)
        self.population_history_prey.append(num_prey)

        # Keep history length bounded
        if len(self.population_history_steps) > self.population_history_max_length:
            self.population_history_steps.pop(0)
            self.population_history_predators.pop(0)
            self.population_history_prey.pop(0)

        self._draw_grid()
        if self.show_fov and (self.predator_obs_range or self.prey_obs_range):
            self._draw_fov_overlays(step_data)
        if walls:  # Draw beneath dynamic entities
            self._draw_walls(walls)
        self._draw_grass(grass_positions, grass_energies)
        self._draw_agents(step_data, agents_just_ate)
        self._draw_legend(step, step_data)
        if self.enable_tooltips:
            self._draw_tooltip(step_data, grass_positions, grass_energies)

        pygame.display.set_caption(f"PredPreyGrass Live Viewer — Step {step}")
        pygame.display.flip()

    def _draw_grid(self):
        self.screen.blit(self.grid_surface, (0, 0))

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

    def _draw_walls(self, walls):
        """Draw static wall cells.

        Accepts set/list of (x,y) or mapping id->(x,y)."""
        if isinstance(walls, dict):
            positions = walls.values()
        else:
            positions = walls
        ml = self.gui_style.margin_left
        mt = self.gui_style.margin_top
        cs = self.cell_size
        for pos in positions:
            x, y = map(int, pos)
            rect = pygame.Rect(ml + x * cs + 1, mt + y * cs + 1, cs - 2, cs - 2)
            pygame.draw.rect(self.screen, self.gui_style.wall_color, rect)

    def _draw_fov_overlays(self, step_data):
        """Draw transparent FOV overlays for each agent.

        Modes:
        - Default (fov_respect_walls=False): draw a solid square region (legacy behaviour).
        - LOS-clipped (fov_respect_walls=True): per-cell line-of-sight test; walls block further cells.
          A wall cell itself is rendered (you can see the wall) but cells behind it along the same line are skipped.
        """
        if self.fov_alpha <= 0:
            return
        # Create a small surface for blending
        cs = self.cell_size
        ml = self.gui_style.margin_left
        mt = self.gui_style.margin_top

        def _range_for_agent(aid):
            if "predator" in aid:
                return self.predator_obs_range
            return self.prey_obs_range

        los_clip = self.fov_respect_walls and bool(self._walls_set)

        for agent_id, data in step_data.items():
            if self.fov_agents is not None and agent_id not in self.fov_agents:
                continue
            obs_r = _range_for_agent(agent_id)
            if not obs_r or obs_r <= 0:
                continue
            pos = tuple(map(int, data["position"]))
            offset = (obs_r - 1) // 2
            x0 = max(0, pos[0] - offset)
            y0 = max(0, pos[1] - offset)
            x1 = min(self.grid_size[0] - 1, pos[0] + offset)
            y1 = min(self.grid_size[1] - 1, pos[1] + offset)

            if "predator" in agent_id:
                base_color = self.fov_color_predator
            else:
                base_color = self.fov_color_prey

            if not los_clip:
                # Fast path: old solid rectangle
                px = ml + x0 * cs
                py = mt + y0 * cs
                w = (x1 - x0 + 1) * cs
                h = (y1 - y0 + 1) * cs
                surf = pygame.Surface((w, h), pygame.SRCALPHA)
                surf.fill(base_color)
                self.screen.blit(surf, (px, py))
                continue

            # LOS-clipped path: iterate cells
            cell_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
            cell_surf.fill(base_color)
            ax, ay = pos

            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    if self._cell_visible_from(ax, ay, cx, cy):
                        px = ml + cx * cs
                        py = mt + cy * cs
                        self.screen.blit(cell_surf, (px, py))

    def _cell_visible_from(self, ax, ay, tx, ty):
        """Return True if target cell (tx,ty) is visible from agent (ax,ay) with wall blocking.

        A wall cell itself is visible; any wall encountered earlier in the ray blocks further cells.
        Bresenham line algorithm adapted for grid; excluding the origin cell from blocking logic.
        """
        if (ax, ay) == (tx, ty):
            return True

        dx = abs(tx - ax)
        dy = abs(ty - ay)
        x, y = ax, ay
        n = 1 + dx + dy
        x_inc = 1 if tx > ax else -1
        y_inc = 1 if ty > ay else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        first = True
        while n > 0:
            if not first and (x, y) in self._walls_set:
                # We hit a wall before reaching target -> target not visible
                if (x, y) != (tx, ty):
                    return False
            if (x, y) == (tx, ty):
                return True
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
            n -= 1
            first = False
        return True

    def _draw_agents(self, step_data, agents_just_ate):
        for agent_id, agent in step_data.items():
            pos = tuple(map(int, agent["position"]))
            energy = agent["energy"]

            # Convert grid position to pixel coordinates
            x_pix = self.gui_style.margin_left + pos[0] * self.cell_size + self.cell_size // 2
            y_pix = self.gui_style.margin_top + pos[1] * self.cell_size + self.cell_size // 2

            # Determine type and type
            if "predator" in agent_id:
                color = self.gui_style.predator_color  # Default predator color
                if "type_1" in agent_id:
                    color = self.gui_style.predator_type_1_color
                elif "type_2" in agent_id:
                    color = self.gui_style.predator_type_2_color
                reference_energy = self.reference_energy_predator

            elif "prey" in agent_id:
                color = self.gui_style.prey_color  # Default prey color
                if "type_1" in agent_id:
                    color = self.gui_style.prey_type_1_color
                elif "type_2" in agent_id:
                    color = self.gui_style.prey_type_2_color
                reference_energy = self.reference_energy_prey

            size_factor = min(energy / reference_energy, 1.0)
            base_radius = self.cell_size // 2 - 2
            radius = int(base_radius * size_factor)

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
            if "energy_reproduction" in agent and agent["energy_reproduction"] < 0:
                pygame.draw.circle(
                    self.screen,
                    self.gui_style.halo_reproduction_color,
                    (x_pix, y_pix),
                    max(radius + 5, 6),
                    width=self.gui_style.halo_reproduction_thickness,
                )

    def _draw_legend(self, step, step_data):
        x = self.gui_style.margin_left + self.grid_size[0] * self.cell_size + 20
        y = self.gui_style.margin_top + 10

        y = self._draw_legend_step_counter(x, y, step)

        y = self._draw_legend_agents(x, y, step_data)

        y = self._draw_legend_environment_elements(x, y)

        if self.enable_speed_slider:
            y = self._draw_legend_type_slider(x, y)

        y = self._draw_legend_population_chart(x, y)

        # type 2 not utilized in this experiment
        # if self._using_type_prefix(step_data):
        #    y = self._draw_legend_type2_percent_chart(x, y)

    def _draw_legend_step_counter(self, x, y, step):
        spacing = self.gui_style.legend_spacing
        font_large = self.font_legend
        step_label_surface = font_large.render("Step:", True, (0, 0, 0))
        self.screen.blit(step_label_surface, (x, y))

        label_width = step_label_surface.get_width()
        step_number_surface = font_large.render(f"{step}", True, (255, 0, 0))
        self.screen.blit(step_number_surface, (x + label_width + 5, y))  # +5 pixels spacing

        return y + spacing

    def _draw_legend_agents(self, x, y, step_data):
        spacing = self.gui_style.legend_spacing
        r = self.gui_style.legend_circle_radius
        font = self.tooltip_font
        font_large = self.font_legend

        # Draw legend title
        title_surface = font_large.render("Agent size depends on energy", True, (0, 0, 0))
        self.screen.blit(title_surface, (x, y))
        y += spacing

        # type 2 not utilized in this experiment
        # is_type_based = self._using_type_prefix(step_data)
        is_type_based = False

        if is_type_based:
            # type-specific predator colors
            pygame.draw.circle(self.screen, self.gui_style.predator_type_1_color, (x + r, y + r), r)
            self.screen.blit(font.render("Predator (type 1)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            pygame.draw.circle(self.screen, self.gui_style.predator_type_2_color, (x + r, y + r), r)
            self.screen.blit(font.render("Predator (type 2)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            # type-specific prey colors
            pygame.draw.circle(self.screen, self.gui_style.prey_type_1_color, (x + r, y + r), r)
            self.screen.blit(font.render("Prey (type 1)", True, (0, 0, 0)), (x + 30, y))
            y += spacing

            pygame.draw.circle(self.screen, self.gui_style.prey_type_2_color, (x + r, y + r), r)
            self.screen.blit(font.render("Prey (type 2)", True, (0, 0, 0)), (x + 30, y))
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
        self.screen.blit(font.render("Reproduction halo", True, (0, 0, 0)), (x + 30, y))
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

        # Wall
        pygame.draw.rect(self.screen, self.gui_style.wall_color, pygame.Rect(x + r - s // 2, y + r - s // 2, s, s))
        self.screen.blit(font.render("Wall", True, (0, 0, 0)), (x + 30, y))
        y += spacing

        if self.show_fov and (self.predator_obs_range or self.prey_obs_range):
            # Predator FOV legend (with alpha blending)
            fov_pred_rect = pygame.Surface((s, s), pygame.SRCALPHA)
            fov_pred_rect.fill(self.fov_color_predator)
            self.screen.blit(fov_pred_rect, (x + r - s // 2, y + r - s // 2))
            self.screen.blit(font.render("Predator FOV", True, (0, 0, 0)), (x + 30, y))
            y += spacing
            # Prey FOV legend (with alpha blending)
            fov_prey_rect = pygame.Surface((s, s), pygame.SRCALPHA)
            fov_prey_rect.fill(self.fov_color_prey)
            self.screen.blit(fov_prey_rect, (x + r - s // 2, y + r - s // 2))
            self.screen.blit(font.render("Prey FOV", True, (0, 0, 0)), (x + 30, y))
            y += spacing

        return y

    def _draw_legend_type_slider(self, x, y):
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
        max_len = self.population_history_max_length
        step = max_len // 5
        x_tick_labels = list(range(0, max_len + 1, step))
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
                pygame.draw.line(self.screen, self.gui_style.predator_type_1_color, (x1, y1p), (x2, y2p), 2)
                y1pr = chart_y + chart_height - int(self.population_history_prey[i - 1] / max_agents * chart_height)
                y2pr = chart_y + chart_height - int(self.population_history_prey[i] / max_agents * chart_height)
                pygame.draw.line(self.screen, self.gui_style.prey_type_1_color, (x1, y1pr), (x2, y2pr), 2)

        return chart_y + chart_height + self.gui_style.legend_spacing

    def _draw_legend_type2_percent_chart(self, x, y):
        chart_width = 260
        chart_height = 100
        chart_x = x + 30
        chart_y = y + 40
        spacing = self.gui_style.legend_spacing

        font_small = pygame.font.SysFont(None, int(self.gui_style.tooltip_font_size * 0.8), bold=False)
        title_surface = font_small.render("type 2 predator and prey population (%)", True, (0, 0, 0))
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
        max_len = self.population_history_max_length
        step = max_len // 5
        x_tick_labels = list(range(0, max_len + 1, step))
        num_xticks = len(x_tick_labels)
        for i, label_value in enumerate(x_tick_labels):
            x_pos = chart_x + int(i / (num_xticks - 1) * chart_width)
            y_pos = chart_y + chart_height + 5
            label_surface = font_small.render(f"{label_value}", True, (0, 0, 0))
            self.screen.blit(label_surface, (x_pos - label_surface.get_width() // 2, y_pos))

        # Draw lines
        for i in range(1, len(self.population_pct_type2_predators)):
            x1 = chart_x + int((i - 1) / self.population_history_max_length * chart_width)
            x2 = chart_x + int(i / self.population_history_max_length * chart_width)

            y1_pred = chart_y + chart_height - int(self.population_pct_type2_predators[i - 1] * chart_height)
            y2_pred = chart_y + chart_height - int(self.population_pct_type2_predators[i] * chart_height)
            pygame.draw.line(self.screen, self.gui_style.predator_type_2_color, (x1, y1_pred), (x2, y2_pred), 2)

            y1_prey = chart_y + chart_height - int(self.population_pct_type2_prey[i - 1] * chart_height)
            y2_prey = chart_y + chart_height - int(self.population_pct_type2_prey[i] * chart_height)
            pygame.draw.line(self.screen, self.gui_style.prey_type_2_color, (x1, y1_prey), (x2, y2_prey), 2)

        return chart_y + chart_height + spacing

    def _draw_tooltip(self, step_data, grass_positions, grass_energies):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x = (mouse_x - self.gui_style.margin_left) // self.cell_size
        grid_y = (mouse_y - self.gui_style.margin_top) // self.cell_size

        # Early exit if mouse is outside grid area
        if not (0 <= grid_x < self.grid_size[0]) or not (0 <= grid_y < self.grid_size[1]):
            return

        hovered_entity = None
        hovered_energy = 0.0

        # Check for hovered agent
        for agent_id, data in step_data.items():
            pos = tuple(map(int, data["position"]))
            if pos == (grid_x, grid_y):
                hovered_entity = agent_id
                hovered_energy = data["energy"]
                break

        # Check for hovered grass (only if no agent found)
        if not hovered_entity:
            for grass_id, pos in grass_positions.items():
                if pos == (grid_x, grid_y):
                    hovered_entity = grass_id
                    hovered_energy = grass_energies.get(grass_id, 0) if grass_energies else 0
                    break

        if hovered_entity:
            font = self.tooltip_font
            lines = []

            # First line: agent or grass ID
            lines.append(("ID", hovered_entity))

            if hovered_entity in step_data:
                agent = step_data[hovered_entity]

                # Second line: unique ID (only for agents, not grass)
                uid = agent.get("unique_id")
                if uid:
                    lines.append(("UID", uid))

                age = agent.get("age")
                if age is not None:
                    lines.append(("Age", f"{age:>6}"))

                # Offspring count from the current live agent only
                offspring = agent.get("offspring_count")
                if offspring is not None:
                    lines.append(("Offspring", f"{offspring:>6}"))

                if "offspring_ids" in agent and agent["offspring_ids"]:
                    for i, uid in enumerate(agent["offspring_ids"]):
                        lines.append((f"Child {i+1}", uid))

                lines.append(("Energy", f"{hovered_energy:+6.2f}"))

                # Energy deltas
                for key, label in [
                    ("energy_decay", "-Decay"),
                    ("energy_movement", "-Movement"),
                    ("energy_eating", "-Eating"),
                    ("energy_reproduction", "-Reproduction"),
                ]:
                    if key in agent:
                        delta = agent[key]
                        lines.append((label, f"{delta:+6.2f}"))

                # Add implicit energy from previous step for debugging
                if all(k in agent for k in ("energy_decay", "energy_movement", "energy_eating", "energy_reproduction")):
                    decay = agent["energy_decay"]
                    move = agent["energy_movement"]
                    eat = agent["energy_eating"]
                    repro = agent["energy_reproduction"]
                    energy_prev = hovered_energy - decay - move - eat - repro
                    lines.append(("Energy_prev", f"{energy_prev:6.2f}"))
            else:
                lines.append(("Energy", f"{hovered_energy:>6.2f}"))

            # Measure width/height
            label_width = max(font.size(label)[0] for label, _ in lines if label)
            value_width = max(font.size(value)[0] for _, value in lines if value)
            line_height = font.get_height()
            padding = self.gui_style.tooltip_padding
            width = label_width + value_width + 12
            height = len(lines) * line_height

            tooltip_x = mouse_x + 10
            tooltip_y = mouse_y + 10

            bg_rect = pygame.Rect(tooltip_x - padding, tooltip_y - padding, width + 2 * padding, height + 2 * padding)
            pygame.draw.rect(self.screen, (255, 255, 200), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)

            # Render lines
            y_offset = tooltip_y
            for label, value in lines:
                if value is None:
                    text_surface = font.render(label, True, (0, 0, 0))
                    self.screen.blit(text_surface, (tooltip_x, y_offset))
                else:
                    label_surface = font.render(f"{label}:", True, (0, 0, 0))
                    value_surface = font.render(value, True, (0, 0, 0))
                    self.screen.blit(label_surface, (tooltip_x, y_offset))
                    self.screen.blit(value_surface, (tooltip_x + label_width + 12, y_offset))
                y_offset += line_height

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
