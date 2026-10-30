"""A pygame viewer for StagHuntContinuous, adapted from
base_environment_continuous's.

Draws the arena (grass and, tinted, the mammoths' steppe; predators and
mammoths as sprites facing their direction of movement, rabbits as small
discs; a ring around every predator whose last join decision was on; the
view cones of one predator and one mammoth), with the three populations
over the episode to its right. It only reads the environment's state and
never draws random numbers, so watching an episode does not change it.

On a machine without a display, set SDL_VIDEODRIVER=dummy to render
off-screen (frames for videos still work).
"""

import math
import os
import subprocess
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
BACKGROUND = (172, 206, 231)
CONE_COLOR = {"predator": (82, 117, 172), "mammoth": (167, 98, 88)}
CONE_ALPHA = 80
POPULATION_COLOR = {"predator": (200, 60, 50), "rabbit": (90, 90, 90), "mammoth": (150, 100, 40)}
RABBIT_COLOR = (235, 235, 225)
JOIN_COLOR = (230, 160, 20)
STEPPE_TINT = (215, 160, 60)
SPRITE_PER_DIAMETER = 1.25  # Sprite size relative to the body diameter.
MIN_SPRITE = 36  # Arena units; keeps small animals recognizable.
GRASS_SPRITE = 24  # Arena units.
# The sprites' own facing: the mammoth looks right, the hunter (predator)
# holds its spear to the left.
SPRITE_FACES_RIGHT = {"predator": False, "mammoth": True}
# Degrees a sprite leans up or down with its movement, at most.
MAX_TILT = {"predator": 20.0, "mammoth": 8.0}
TITLE_BAR = 40  # Room left for the window's title bar, in pixels.


def sprite_pose(vx, vy, previous=(True, 0.0), max_tilt=20.0):
    """(faces_right, tilt) for an animal moving with velocity (vx, vy) in
    screen coordinates (y down): it faces the way it moves horizontally and
    leans up or down by the vertical part of its movement, at most max_tilt
    degrees, so it never turns upside down. A stopped animal keeps its
    previous pose; straight up or down keeps the previous facing."""
    if math.hypot(vx, vy) < 1e-6:
        return previous
    right = previous[0] if abs(vx) < 1e-6 else vx > 0
    tilt = math.degrees(math.atan2(-vy, abs(vx)))
    return right, max(-max_tilt, min(max_tilt, tilt))


def work_area(pygame):
    """The screen area not taken by panels or task bars, as (x, y, width,
    height): the window manager's _NET_WORKAREA on X11, else the desktop
    minus a typical task bar; None without a screen."""
    try:
        out = subprocess.run(
            ["xprop", "-root", "_NET_WORKAREA"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        x, y, width, height = (int(v) for v in out.split("=")[1].split(",")[:4])
        if width > 0 and height > 0:
            return x, y, width, height
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    desktop = pygame.display.Info()
    if desktop.current_w <= 0 or desktop.current_h <= 0:
        return None
    return 0, 0, desktop.current_w, desktop.current_h - 60


class Viewer:
    """Draws an environment's current state; call reset() after the
    environment's reset() and draw() after it and after every step().

    window_size: "fit" (the largest arena that fits the screen, with an
    equally wide population chart beside it), the arena's longer side in
    pixels, or None (one pixel per arena unit). fps: the highest frame rate
    of the window (None: as fast as possible).
    """

    def __init__(self, env, window_size="fit", fps=None, draw_view_cones=True, caption="PredPreyGrass stag hunt"):
        import pygame

        if window_size not in ("fit", None) and (
            isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1
        ):
            raise ValueError('window_size must be "fit", None, or pixels')
        self.pygame = pygame
        self.env = env
        self.fps = fps
        self.draw_view_cones = draw_view_cones
        pygame.init()
        headless = pygame.display.get_driver() in ("dummy", "offscreen")
        area = None if headless else work_area(pygame)
        arena = (env.width, env.height)
        if window_size is None or (window_size == "fit" and area is None):
            scale = 1.0
        elif window_size == "fit":
            _, _, width, height = area
            scale = min(width / (2 * arena[0]), (height - TITLE_BAR) / arena[1])
        else:
            scale = window_size / max(arena)
        self.arena_px = (round(arena[0] * scale), round(arena[1] * scale))
        total = (2 * self.arena_px[0], self.arena_px[1])
        if area is not None:
            # Top of the work area, centred horizontally; the window manager
            # keeps the title bar on screen. Left alone if the user set it.
            x, y, width, _ = area
            left = x + max(0, (width - total[0]) // 2)
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", f"{left},{y}")
        pygame.display.set_caption(caption)
        self.window = pygame.display.set_mode(total)
        self.canvas = pygame.Surface(arena)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, max(16, self.arena_px[1] // 40))
        self.title_font = pygame.font.Font(None, max(20, self.arena_px[1] // 28))
        self.sprites = {species: self._sprite(species) for species in ("predator", "mammoth")}
        grass = pygame.image.load(str(ASSETS / "grass-transparent.png")).convert_alpha()
        grass = grass.subsurface(grass.get_bounding_rect(min_alpha=1))
        height = max(1, round(GRASS_SPRITE * grass.get_height() / grass.get_width()))
        self.grass_sprite = pygame.transform.smoothscale(grass, (GRASS_SPRITE, height))
        self.steppe_sprite = self.grass_sprite.copy()
        self.steppe_sprite.fill(STEPPE_TINT + (255,), special_flags=pygame.BLEND_RGBA_MULT)
        self.poses = {}  # agent: (faces_right, tilt)
        self.watched = {"predator": None, "mammoth": None}  # whose cone is drawn
        self.history = []  # (physics step, predators, rabbits, mammoths)
        self.closed = False

    def _sprite(self, species):
        """The species' image, cropped to its visible part and scaled so its
        longer side is SPRITE_PER_DIAMETER body diameters (at least
        MIN_SPRITE arena units)."""
        pygame = self.pygame
        image = pygame.image.load(str(ASSETS / f"{species}.png")).convert_alpha()
        image = image.subsurface(image.get_bounding_rect(min_alpha=1))
        radius = self.env.body[species][0]
        target = max(SPRITE_PER_DIAMETER * 2 * radius, MIN_SPRITE)
        scale = target / max(image.get_size())
        size = (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale)))
        return pygame.transform.smoothscale(image, size)

    def _copies(self, x, y):
        """A position and its images one arena width/height away, so things
        straddling an edge show on both sides."""
        for dx in (-self.env.width, 0, self.env.width):
            for dy in (-self.env.height, 0, self.env.height):
                yield x + dx, y + dy

    def reset(self):
        """Forget the previous episode: its population history, the
        animals' poses and whose view cones are shown."""
        self.history.clear()
        self.poses.clear()
        self.watched = {"predator": None, "mammoth": None}

    def draw(self):
        """Draw the current state and show it; returns the window as an RGB
        array of shape (height, width, 3)."""
        pygame = self.pygame
        env = self.env
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
        if self.history and env.time_step < self.history[-1][0]:
            self.reset()  # a new episode without reset()
        self.history.append((env.time_step, len(env.predators), len(env.rabbits), len(env.mammoths)))
        self.canvas.fill(BACKGROUND)
        for layer, sprite in (("grass", self.grass_sprite), ("steppe", self.steppe_sprite)):
            food = env.food[layer]
            for patch in food.patches:
                if food.available(patch):
                    for x, y in self._copies(patch.position.x, patch.position.y):
                        self.canvas.blit(sprite, sprite.get_rect(center=(round(x), round(y))))
        if self.draw_view_cones:
            self._draw_cones()
        for animal in env.mammoths + env.rabbits + env.predators:
            self._draw_animal(animal)
        arena = self.window.subsurface((0, 0, *self.arena_px))
        pygame.transform.smoothscale(self.canvas, self.arena_px, arena)
        panel = self.window.subsurface((self.arena_px[0], 0, self.window.get_width() - self.arena_px[0], self.arena_px[1]))
        self._draw_population(panel)
        pygame.display.update()
        if self.fps:
            self.clock.tick(self.fps)
        return np.transpose(pygame.surfarray.array3d(self.window), (1, 0, 2))

    def _draw_cones(self):
        """The view cone of one predator and one mammoth, each followed
        until it dies."""
        pygame = self.pygame
        overlay = pygame.Surface(self.canvas.get_size(), pygame.SRCALPHA)
        for species, animals in (("predator", self.env.predators), ("mammoth", self.env.mammoths)):
            if self.watched[species] not in animals:
                self.watched[species] = animals[0] if animals else None
            animal = self.watched[species]
            if animal is None:
                continue
            # Orientation is counterclockwise with y down; screen angles
            # are clockwise.
            centre = -animal.orientation
            # Points every degree or less along the arc, ends included.
            steps = max(2, math.ceil(animal.fov))
            angles = [math.radians(centre - animal.fov / 2 + animal.fov * i / steps) for i in range(steps + 1)]
            for x, y in self._copies(animal.position.x, animal.position.y):
                points = [(x, y)] + [
                    (x + animal.view_distance * math.cos(angle), y + animal.view_distance * math.sin(angle)) for angle in angles
                ]
                pygame.draw.polygon(overlay, CONE_COLOR[species] + (CONE_ALPHA,), points)
        self.canvas.blit(overlay, (0, 0))

    def _draw_animal(self, animal):
        pygame = self.pygame
        species = animal.species
        if species == "rabbit":
            for x, y in self._copies(animal.position.x, animal.position.y):
                pygame.draw.circle(self.canvas, RABBIT_COLOR, (round(x), round(y)), round(animal.radius * 0.8))
                pygame.draw.circle(self.canvas, (120, 120, 110), (round(x), round(y)), round(animal.radius * 0.8), 2)
            return
        if species == "predator" and self.env.last_joins.get(animal.id):
            for x, y in self._copies(animal.position.x, animal.position.y):
                pygame.draw.circle(self.canvas, JOIN_COLOR, (round(x), round(y)), round(animal.radius * 1.6), 3)
        right, tilt = sprite_pose(
            animal.velocity.x,
            animal.velocity.y,
            self.poses.get(animal.id, (SPRITE_FACES_RIGHT[species], 0.0)),
            MAX_TILT[species],
        )
        self.poses[animal.id] = (right, tilt)
        image = self.sprites[species]
        if right != SPRITE_FACES_RIGHT[species]:
            image = pygame.transform.flip(image, True, False)
        # pygame rotates counterclockwise: a right-facing sprite leans up with
        # a positive angle, a left-facing one with a negative angle.
        image = pygame.transform.rotate(image, tilt if right else -tilt)
        for x, y in self._copies(animal.position.x, animal.position.y):
            self.canvas.blit(image, (x - image.get_width() // 2, y - image.get_height() // 2))

    def _draw_population(self, surface):
        """Predator, rabbit and mammoth counts over the episode: physics steps on the x
        axis, animals on the y axis."""
        pygame = self.pygame
        width, height = surface.get_size()
        surface.fill((250, 250, 248))
        if width < 100:
            return
        max_steps = self.env.max_time_steps
        ink, grid = (60, 60, 60), (225, 225, 220)
        left, right = int(width * 0.11), int(width * 0.04)
        top, bottom = int(height * 0.09), int(height * 0.08)
        plot_w, plot_h = width - left - right, height - top - bottom
        peak = max([1] + [max(row[1:]) for row in self.history])
        y_max = next((m for m in (10, 20, 25, 40, 50, 75, 100, 150, 200, 500) if m >= peak), peak)

        def to_xy(step, count):
            x = left + plot_w * min(step, max_steps) / max_steps
            return x, top + plot_h * (1 - count / y_max)

        for i in range(6):  # gridlines and axis labels
            count = y_max * i / 5
            _, y = to_xy(0, count)
            pygame.draw.line(surface, grid, (left, y), (left + plot_w, y))
            label = self.font.render(f"{count:g}", True, ink)
            surface.blit(label, (left - label.get_width() - 6, y - label.get_height() / 2))
            step = max_steps * i / 5
            x, _ = to_xy(step, 0)
            label = self.font.render(f"{step:g}", True, ink)
            surface.blit(label, (x - label.get_width() / 2, top + plot_h + 6))
        pygame.draw.rect(surface, ink, (left, top, plot_w, plot_h), 1)
        for index, species in ((1, "predator"), (2, "rabbit"), (3, "mammoth")):
            points = [to_xy(row[0], row[index]) for row in self.history]
            if len(points) > 1:
                pygame.draw.lines(surface, POPULATION_COLOR[species], False, points, 3)
        counts = dict(zip(("predator", "rabbit", "mammoth"), self.history[-1][1:] if self.history else (0, 0, 0)))
        surface.blit(self.title_font.render("Population", True, ink), (left, top * 0.3))
        legend_x = left + plot_w
        for text, species in (
            (f"mammoths {counts['mammoth']}", "mammoth"),
            (f"rabbits {counts['rabbit']}", "rabbit"),
            (f"predators {counts['predator']}", "predator"),
        ):
            label = self.font.render(text, True, POPULATION_COLOR[species])
            legend_x -= label.get_width()
            surface.blit(label, (legend_x, top * 0.45))
            legend_x -= 24
        caption = self.font.render("physics step", True, ink)
        surface.blit(caption, (left + plot_w / 2 - caption.get_width() / 2, height - bottom * 0.45))

    def close(self):
        self.pygame.display.quit()
