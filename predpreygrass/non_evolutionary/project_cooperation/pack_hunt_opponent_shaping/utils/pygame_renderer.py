"""Minimal pygame renderer for pack_hunt_opponent_shaping.

Deliberately not built on the full-ecology renderers in the sibling
project_cooperation modules (grass, walls, generations, FOV, ...) -- this
environment has none of that. Shows the grid, predators (numbered, colored by
whether they're engaged this step), the prey, the two radii around the prey,
and a small HUD (step, round, round timeout progress, last event).
"""

import pygame

PREDATOR_COLOR = (200, 30, 30)
PREDATOR_ENGAGED_COLOR = (255, 140, 0)
PREY_COLOR = (30, 60, 220)
GRID_COLOR = (210, 210, 210)
BACKGROUND_COLOR = (250, 250, 250)
TEXT_COLOR = (20, 20, 20)
CAPTURE_RING_COLOR = (255, 0, 0)
ENGAGEMENT_RING_COLOR = (255, 170, 0)
CATCH_FLASH_COLOR = (0, 170, 0)
ESCAPE_FLASH_COLOR = (120, 120, 120)

HUD_HEIGHT = 110
FLASH_DURATION_STEPS = 6


class PackHuntRenderer:
    def __init__(self, grid_size, cell_size=48, target_fps=8):
        pygame.init()
        pygame.font.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.target_fps = target_fps
        width = grid_size * cell_size
        height = grid_size * cell_size + HUD_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Pack Hunt Opponent Shaping")
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 14)

    def _cell_center(self, row, col):
        return (
            col * self.cell_size + self.cell_size // 2,
            HUD_HEIGHT + row * self.cell_size + self.cell_size // 2,
        )

    def _draw_grid(self):
        width = self.grid_size * self.cell_size
        for i in range(self.grid_size + 1):
            x = i * self.cell_size
            pygame.draw.line(self.screen, GRID_COLOR, (x, HUD_HEIGHT), (x, HUD_HEIGHT + width))
            y = HUD_HEIGHT + i * self.cell_size
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (width, y))

    def _draw_radius_ring(self, center, radius_cells, color):
        radius_px = int(radius_cells * self.cell_size)
        pygame.draw.circle(self.screen, color, center, radius_px, width=1)

    def update(self, env):
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_grid()

        # Freeze on the true capture/escape spot for exactly one frame (the
        # event frame itself) instead of jumping straight to the next
        # round's fresh prey position -- the environment always respawns
        # immediately (observations stay correct/up to date), this is a
        # display-only freeze-frame. Later flash frames (border color only,
        # below) show live state, so predators already chasing the new prey
        # don't appear to be walking away from a stale marker.
        steps_since_event = env.current_step - env.last_event_step
        flashing = env.last_event is not None and steps_since_event <= FLASH_DURATION_STEPS
        on_event_frame = env.last_event is not None and steps_since_event == 0
        prey_pos = env.last_event_position if (on_event_frame and env.last_event_position is not None) else env.prey_position

        prey_center = self._cell_center(*prey_pos)
        self._draw_radius_ring(prey_center, env.engagement_radius, ENGAGEMENT_RING_COLOR)
        self._draw_radius_ring(prey_center, env.capture_radius, CAPTURE_RING_COLOR)
        pygame.draw.circle(self.screen, PREY_COLOR, prey_center, self.cell_size // 3)

        for i, agent_id in enumerate(env.agents):
            pos = env.predator_positions[agent_id]
            center = self._cell_center(*pos)
            engaged = env.engaged_this_step.get(agent_id, False)
            color = PREDATOR_ENGAGED_COLOR if engaged else PREDATOR_COLOR
            pygame.draw.circle(self.screen, color, center, self.cell_size // 3)
            label = self.small_font.render(str(i), True, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=center))

        if flashing:
            flash_color = CATCH_FLASH_COLOR if env.last_event == "catch" else ESCAPE_FLASH_COLOR
            width = self.grid_size * self.cell_size
            height = self.grid_size * self.cell_size
            pygame.draw.rect(self.screen, flash_color, (0, HUD_HEIGHT, width, height), width=4)

        self._draw_hud(env)
        pygame.display.flip()

    def _draw_hud(self, env):
        pygame.draw.rect(self.screen, (235, 235, 235), (0, 0, self.grid_size * self.cell_size, HUD_HEIGHT))
        n_engaged = sum(env.engaged_this_step.values())
        event_str = env.last_event if env.last_event is not None else "-"
        line1 = f"step {env.current_step}/{env.max_episode_steps}  round {env.round_index}"
        line2 = f"round_step {env.round_step}/{env.round_timeout_steps}  engaged {n_engaged}/{env.n_predators}"
        line3 = f"last event: {event_str}"
        line4 = "orange=engaged  rings=engagement/capture"
        self.screen.blit(self.font.render(line1, True, TEXT_COLOR), (8, 4))
        self.screen.blit(self.small_font.render(line2, True, TEXT_COLOR), (8, 26))
        self.screen.blit(self.small_font.render(line3, True, TEXT_COLOR), (8, 46))
        self.screen.blit(self.small_font.render(line4, True, TEXT_COLOR), (8, 66))

    def close(self):
        pygame.quit()
