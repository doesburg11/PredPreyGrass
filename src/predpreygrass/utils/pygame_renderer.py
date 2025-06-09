import pygame
from dataclasses import dataclass


@dataclass
class GuiStyle:
    margin_left: int = 10
    margin_top: int = 10
    margin_right: int = 280
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
    grid_color: tuple = (200, 200, 200)
    background_color: tuple = (255, 255, 255)
    halo_reproduction_color: tuple = (255, 0, 0)  # Gold
    halo_eating_color: tuple = (0, 128, 0)  # Bright green
    halo_reproduction_thickness: int = 3
    halo_eating_thickness: int = 3


class PyGameRenderer:
    def __init__(self, grid_size, cell_size=32):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.gui_style = GuiStyle()

        window_width = self.gui_style.margin_left + grid_size[0] * cell_size + self.gui_style.margin_right
        window_height = self.gui_style.margin_top + grid_size[1] * cell_size + self.gui_style.margin_bottom
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("PredPreyGrass Live Viewer")

        self.font = pygame.font.SysFont(None, int(cell_size * 0.5))
        self.tooltip_font = pygame.font.SysFont(None, self.gui_style.tooltip_font_size)

        self.reference_energy_predator = 10.0
        self.reference_energy_prey = 3.0
        self.reference_energy_grass = 2.0

        self.halo_trigger_fraction = 0.9
        self.predator_creation_energy_threshold = 12.0
        self.prey_creation_energy_threshold = 8.0

        self.previous_agent_energies = {}

    def update(
        self,
        agent_positions,
        grass_positions,
        agent_energies=None,
        grass_energies=None,
        step=0,
        agents_just_ate=None
    ):
        if agents_just_ate is None:
            agents_just_ate = set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill(self.gui_style.background_color)

        self._draw_grid()
        self._draw_grass(grass_positions, grass_energies)
        self._draw_agents(agent_positions, agent_energies, agents_just_ate)
        self._draw_tooltip(agent_positions, grass_positions, agent_energies, grass_energies)
        self._draw_legend()

        pygame.display.set_caption(f"PredPreyGrass Live Viewer — Step {step}")
        pygame.display.flip()

    def _draw_grid(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(
                    self.gui_style.margin_left + x * self.cell_size,
                    self.gui_style.margin_top + y * self.cell_size,
                    self.cell_size, self.cell_size
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
            rect = pygame.Rect(
                x_pix - rect_size // 2,
                y_pix - rect_size // 2,
                rect_size,
                rect_size
            )
            pygame.draw.rect(self.screen, color, rect)

    def _draw_agents(self, agent_positions, agent_energies, agents_just_ate):
        for agent_id, pos in agent_positions.items():
            x_pix = self.gui_style.margin_left + pos[0] * self.cell_size + self.cell_size // 2
            y_pix = self.gui_style.margin_top + pos[1] * self.cell_size + self.cell_size // 2
            energy = agent_energies.get(agent_id, 0) if agent_energies else 0

            if "predator" in agent_id:
                color = self.gui_style.predator_color
                size_factor = min(energy / self.reference_energy_predator, 1.0)
                threshold = self.predator_creation_energy_threshold
            elif "prey" in agent_id:
                color = self.gui_style.prey_color
                size_factor = min(energy / self.reference_energy_prey, 1.0)
                threshold = self.prey_creation_energy_threshold
            else:
                color = (0, 0, 0)
                size_factor = 1.0
                threshold = None

            base_radius = self.cell_size // 2 - 2
            radius = int(base_radius * size_factor)

            pygame.draw.circle(self.screen, color, (x_pix, y_pix), max(radius, 2))

            # Draw green ring if agent.just_ate
            if agent_id in agents_just_ate:
                pygame.draw.circle(
                    self.screen,
                    self.gui_style.halo_eating_color,  # Bright green
                    (x_pix, y_pix),
                    max(radius + 5, 6),
                    width=self.gui_style.halo_eating_thickness
                )

            # Draw reproduction halo
            if threshold and energy >= threshold * self.halo_trigger_fraction:
                pygame.draw.circle(
                    self.screen,
                    self.gui_style.halo_reproduction_color,
                    (x_pix, y_pix),
                    max(radius + 5, 6),
                    width=self.gui_style.halo_reproduction_thickness
                )

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
            tooltip_line2 = self.tooltip_font.render(f"E: {hovered_energy:.2f}", True, (0, 0, 0))
            padding = self.gui_style.tooltip_padding
            width = max(tooltip_line1.get_width(), tooltip_line2.get_width())
            height = tooltip_line1.get_height() + tooltip_line2.get_height()
            tooltip_x = mouse_x + 10
            tooltip_y = mouse_y + 10
            bg_rect = pygame.Rect(
                tooltip_x - padding,
                tooltip_y - padding,
                width + 2 * padding,
                height + 2 * padding
            )
            pygame.draw.rect(self.screen, (255, 255, 200), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)
            self.screen.blit(tooltip_line1, (tooltip_x, tooltip_y))
            self.screen.blit(tooltip_line2, (tooltip_x, tooltip_y + tooltip_line1.get_height()))

    def _draw_legend(self):
        x = self.gui_style.margin_left + self.grid_size[0] * self.cell_size + 20
        y = self.gui_style.margin_top + 10
        spacing = self.gui_style.legend_spacing
        r = self.gui_style.legend_circle_radius
        s = self.gui_style.legend_square_size
        font = self.tooltip_font

        # Draw legend title
        title_surface = font.render("Agent size energy related", True, (0, 0, 0))
        self.screen.blit(title_surface, (x, y))

        y += spacing  # Move down after title
        pygame.draw.circle(self.screen, self.gui_style.predator_color, (x + r, y + r), r)
        self.screen.blit(font.render("Predator", True, (0, 0, 0)), (x + 30, y))

        y += spacing
        pygame.draw.circle(self.screen, self.gui_style.prey_color, (x + r, y + r), r)
        self.screen.blit(font.render("Prey", True, (0, 0, 0)), (x + 30, y))

        y += spacing
        pygame.draw.rect(self.screen, self.gui_style.grass_color, pygame.Rect(x + r - s // 2, y + r - s // 2, s, s))
        self.screen.blit(font.render("Grass", True, (0, 0, 0)), (x + 30, y))

        y += spacing
        pygame.draw.circle(
            self.screen,
            self.gui_style.halo_reproduction_color,
            (x + r, y + r),
            r + 2,
            width=self.gui_style.halo_reproduction_thickness
        )
        self.screen.blit(font.render("Reproduction halo", True, (0, 0, 0)), (x + 30, y))

        y += spacing
        pygame.draw.circle(
            self.screen,
            self.gui_style.halo_eating_color,
            (x + r, y + r),
            r + 2,
            width=self.gui_style.halo_eating_thickness
        )
        self.screen.blit(font.render("Eating halo", True, (0, 0, 0)), (x + 30, y))

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
