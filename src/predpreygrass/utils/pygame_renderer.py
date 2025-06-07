import pygame


class PyGameRenderer:
    def __init__(self, grid_size, cell_size=32):
        """
        grid_size: (width, height) tuple (same as env.grid_size, env.grid_size)
        cell_size: size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size

        pygame.init()
        window_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("PredPreyGrass Live Viewer")

        # Font for future agent text (if needed again)
        self.font = pygame.font.SysFont(None, int(cell_size * 0.5))

        # Tooltip font (independent size)
        self.tooltip_font = pygame.font.SysFont(None, 28)

        # Reference energies for size scaling (defaults from your env)
        self.reference_energy_predator = 10.0
        self.reference_energy_prey = 3.0
        self.reference_energy_grass = 2.0

    def update(self, agent_positions, grass_positions, agent_energies=None, grass_energies=None, step=0):
        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw grass first
        for grass_id, pos in grass_positions.items():
            x_pix = pos[0] * self.cell_size + self.cell_size // 2
            y_pix = pos[1] * self.cell_size + self.cell_size // 2

            color = (0, 128, 0)

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

        # Draw agents (predators, prey) — energy-based circle size
        for agent_id, pos in agent_positions.items():
            x_pix = pos[0] * self.cell_size + self.cell_size // 2
            y_pix = pos[1] * self.cell_size + self.cell_size // 2

            energy = agent_energies.get(agent_id, 0) if agent_energies else 0

            if "predator" in agent_id:
                color = (255, 0, 0)
                size_factor = min(energy / self.reference_energy_predator, 1.0)
            elif "prey" in agent_id:
                color = (0, 0, 255)
                size_factor = min(energy / self.reference_energy_prey, 1.0)
            else:
                color = (0, 0, 0)
                size_factor = 1.0  # fallback

            base_radius = self.cell_size // 2 - 2
            radius = int(base_radius * size_factor)

            pygame.draw.circle(self.screen, color, (x_pix, y_pix), max(radius, 2))

        # TOOLTIP: show agent info if mouse over agent
        mouse_x, mouse_y = pygame.mouse.get_pos()
        grid_x = mouse_x // self.cell_size
        grid_y = mouse_y // self.cell_size

        hovered_entity = None
        hovered_energy = 0.0

        # First check agent_positions
        for agent_id, pos in agent_positions.items():
            if pos == (grid_x, grid_y):
                hovered_entity = agent_id
                hovered_energy = agent_energies.get(agent_id, 0) if agent_energies else 0
                break

        # If no agent found, check grass_positions
        if not hovered_entity:
            for grass_id, pos in grass_positions.items():
                if pos == (grid_x, grid_y):
                    hovered_entity = grass_id
                    hovered_energy = grass_energies.get(grass_id, 0) if grass_energies else 0
                    break

        # Draw tooltip if any entity found
        if hovered_entity:
            tooltip_line1 = self.tooltip_font.render(f"{hovered_entity}", True, (0, 0, 0))
            tooltip_line2 = self.tooltip_font.render(f"E: {hovered_energy:.2f}", True, (0, 0, 0))

            tooltip_padding = 4
            width = max(tooltip_line1.get_width(), tooltip_line2.get_width())
            height = tooltip_line1.get_height() + tooltip_line2.get_height()

            tooltip_x = mouse_x + 10
            tooltip_y = mouse_y + 10

            bg_rect = pygame.Rect(
                tooltip_x - tooltip_padding,
                tooltip_y - tooltip_padding,
                width + 2 * tooltip_padding,
                height + 2 * tooltip_padding
            )

            pygame.draw.rect(self.screen, (255, 255, 200), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 1)

            self.screen.blit(tooltip_line1, (tooltip_x, tooltip_y))
            self.screen.blit(tooltip_line2, (tooltip_x, tooltip_y + tooltip_line1.get_height()))

        # Update window title
        pygame.display.set_caption(f"PredPreyGrass Live Viewer — Step {step}")
        pygame.display.flip()

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
