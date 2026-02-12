import pygame
from dataclasses import dataclass


@dataclass
class GuiStyle:
    margin: int = 10
    panel_width: int = 260
    panel_padding: int = 12
    background_color: tuple = (245, 245, 245)
    panel_background: tuple = (235, 235, 235)
    grid_color: tuple = (210, 210, 210)
    predator_color: tuple = (220, 60, 60)
    prey_color: tuple = (60, 90, 220)
    grass_color: tuple = (50, 160, 70)
    text_color: tuple = (20, 20, 20)
    line_predator: tuple = (220, 60, 60)
    line_prey: tuple = (60, 90, 220)


class PyGameRenderer:
    def __init__(self, width: int, height: int, cell_size: int = 16, fps: int = 20):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps
        self.style = GuiStyle()

        window_width = self.style.margin * 2 + width * cell_size + self.style.panel_width
        window_height = self.style.margin * 2 + height * cell_size + 24
        pygame.init()
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Minimal Ecology Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, max(12, int(cell_size * 0.9)))
        self.small_font = pygame.font.SysFont(None, max(12, int(cell_size * 0.75)))

        self.history_steps = []
        self.history_prey = []
        self.history_pred = []
        self.history_max = 200

    def close(self) -> None:
        pygame.quit()

    def update(self, state, config, step: int, stats: dict | None = None) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(self.style.background_color)
        self._draw_grass(state, config)
        self._draw_grid()
        self._draw_agents(state, config)
        self._draw_text(state, step)
        self._draw_panel(state, step, stats)

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _draw_grid(self) -> None:
        for x in range(self.width + 1):
            x_pix = self.style.margin + x * self.cell_size
            pygame.draw.line(
                self.screen,
                self.style.grid_color,
                (x_pix, self.style.margin),
                (x_pix, self.style.margin + self.height * self.cell_size),
                1,
            )
        for y in range(self.height + 1):
            y_pix = self.style.margin + y * self.cell_size
            pygame.draw.line(
                self.screen,
                self.style.grid_color,
                (self.style.margin, y_pix),
                (self.style.margin + self.width * self.cell_size, y_pix),
                1,
            )

    def _draw_grass(self, state, config) -> None:
        gmax = max(config.gmax, 1e-6)
        for y in range(self.height):
            for x in range(self.width):
                amount = state.grass[y][x]
                if amount <= 0:
                    continue
                intensity = min(amount / gmax, 1.0)
                color = (
                    int(self.style.grass_color[0] * intensity),
                    int(self.style.grass_color[1] * intensity),
                    int(self.style.grass_color[2] * intensity),
                )
                rect = pygame.Rect(
                    self.style.margin + x * self.cell_size,
                    self.style.margin + y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

    def _draw_agents(self, state, config) -> None:
        for agent in state.agents:
            x_pix = self.style.margin + agent.x * self.cell_size + self.cell_size // 2
            y_pix = self.style.margin + agent.y * self.cell_size + self.cell_size // 2
            if agent.kind == "predator":
                color = self.style.predator_color
                ref_energy = config.predator_energy_init
            else:
                color = self.style.prey_color
                ref_energy = config.prey_energy_init
            size_factor = min(agent.energy / max(ref_energy, 1.0), 1.0)
            radius = max(2, int((self.cell_size // 2 - 1) * size_factor))
            pygame.draw.circle(self.screen, color, (x_pix, y_pix), radius)

    def _draw_text(self, state, step: int) -> None:
        prey = sum(1 for agent in state.agents if agent.kind == "prey")
        pred = sum(1 for agent in state.agents if agent.kind == "predator")
        text = f"t={step} prey={prey} pred={pred}"
        surface = self.font.render(text, True, self.style.text_color)
        self.screen.blit(surface, (self.style.margin, self.style.margin + self.height * self.cell_size + 2))

    def _draw_panel(self, state, step: int, stats: dict | None) -> None:
        panel_x = self.style.margin + self.width * self.cell_size + self.style.margin
        panel_y = self.style.margin
        panel_w = self.style.panel_width - self.style.margin
        panel_h = self.height * self.cell_size
        rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, self.style.panel_background, rect)

        prey = sum(1 for agent in state.agents if agent.kind == "prey")
        pred = sum(1 for agent in state.agents if agent.kind == "predator")
        self._push_history(step, prey, pred)

        y = panel_y + self.style.panel_padding
        y = self._draw_panel_line(panel_x, y, f"Step: {step}")
        y = self._draw_panel_line(panel_x, y, f"Prey: {prey}")
        y = self._draw_panel_line(panel_x, y, f"Pred: {pred}")

        if stats:
            grass = stats.get("grass", {}).get("mean", None)
            if grass is not None:
                y = self._draw_panel_line(panel_x, y, f"Grass: {grass:.2f}")

            prey_stats = stats.get("prey", {})
            pred_stats = stats.get("predator", {})
            y += 6
            y = self._draw_panel_line(panel_x, y, "Prey traits (mean):", bold=True)
            y = self._draw_panel_trait(panel_x, y, prey_stats, "speed")
            y = self._draw_panel_trait(panel_x, y, prey_stats, "vision")
            y = self._draw_panel_trait(panel_x, y, prey_stats, "metabolism")
            y += 6
            y = self._draw_panel_line(panel_x, y, "Pred traits (mean):", bold=True)
            y = self._draw_panel_trait(panel_x, y, pred_stats, "speed")
            y = self._draw_panel_trait(panel_x, y, pred_stats, "vision")
            y = self._draw_panel_trait(panel_x, y, pred_stats, "attack_power")
            y = self._draw_panel_trait(panel_x, y, pred_stats, "metabolism")

        # Sparkline at bottom
        spark_h = 90
        spark_y = panel_y + panel_h - spark_h - self.style.panel_padding
        spark_rect = pygame.Rect(panel_x + self.style.panel_padding, spark_y, panel_w - 2 * self.style.panel_padding, spark_h)
        pygame.draw.rect(self.screen, (225, 225, 225), spark_rect)
        self._draw_sparkline(spark_rect)

    def _push_history(self, step: int, prey: int, pred: int) -> None:
        self.history_steps.append(step)
        self.history_prey.append(prey)
        self.history_pred.append(pred)
        if len(self.history_steps) > self.history_max:
            self.history_steps.pop(0)
            self.history_prey.pop(0)
            self.history_pred.pop(0)

    def _draw_panel_line(self, x: int, y: int, text: str, bold: bool = False) -> int:
        font = self.font if bold else self.small_font
        surface = font.render(text, True, self.style.text_color)
        self.screen.blit(surface, (x + self.style.panel_padding, y))
        return y + surface.get_height() + 2

    def _draw_panel_trait(self, x: int, y: int, stats: dict, name: str) -> int:
        key = f"{name}_mean"
        value = stats.get(key)
        if value is None:
            return y
        return self._draw_panel_line(x, y, f"  {name}: {value:.2f}")

    def _draw_sparkline(self, rect: pygame.Rect) -> None:
        if len(self.history_steps) < 2:
            return
        max_count = max(max(self.history_prey), max(self.history_pred), 1)
        n = len(self.history_steps)
        for i in range(1, n):
            x0 = rect.x + int((i - 1) / (n - 1) * rect.width)
            x1 = rect.x + int(i / (n - 1) * rect.width)

            y0 = rect.y + rect.height - int(self.history_prey[i - 1] / max_count * rect.height)
            y1 = rect.y + rect.height - int(self.history_prey[i] / max_count * rect.height)
            pygame.draw.line(self.screen, self.style.line_prey, (x0, y0), (x1, y1), 2)

            y0 = rect.y + rect.height - int(self.history_pred[i - 1] / max_count * rect.height)
            y1 = rect.y + rect.height - int(self.history_pred[i] / max_count * rect.height)
            pygame.draw.line(self.screen, self.style.line_predator, (x0, y0), (x1, y1), 2)
