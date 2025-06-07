import pygame


class SimplePyGameRenderer:
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

        self.font = pygame.font.SysFont(None, int(cell_size * 0.5))

    def update(self, agent_positions, grass_positions, agent_energies=None, step=0):
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

            # Grass is green square
            color = (0, 128, 0)
            rect_size = self.cell_size * 0.8
            rect = pygame.Rect(
                x_pix - rect_size // 2,
                y_pix - rect_size // 2,
                rect_size,
                rect_size
            )
            pygame.draw.rect(self.screen, color, rect)

        # Draw agents
        for agent_id, pos in agent_positions.items():
            x_pix = pos[0] * self.cell_size + self.cell_size // 2
            y_pix = pos[1] * self.cell_size + self.cell_size // 2

            # Determine color
            if "predator" in agent_id:
                color = (255, 0, 0)
            elif "prey" in agent_id:
                color = (0, 0, 255)
            else:
                color = (0, 0, 0)

            # Draw agent circle
            radius = self.cell_size // 2 - 2
            pygame.draw.circle(self.screen, color, (x_pix, y_pix), radius)

            # Draw energy if available
            text_y_offset = -radius + 2
            if agent_energies and agent_id in agent_energies:
                energy_text = self.font.render(f"E:{agent_energies[agent_id]:.1f}", True, (0, 0, 0))
                self.screen.blit(energy_text, (x_pix - radius, y_pix + text_y_offset))
                text_y_offset += self.font.get_height()

        # Update window title
        pygame.display.set_caption(f"PredPreyGrass Live Viewer — Step {step}")
        pygame.display.flip()

    def close(self):
        pygame.quit()
