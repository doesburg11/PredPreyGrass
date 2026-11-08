"""2D vectors and torus geometry.

The arithmetic deliberately follows the order of operations of the
environment this module replaces (Aquarium 0.1.10 with add_walls's patches),
so that results are identical bit for bit (SPEC.md, section 10).
"""

import math


class Vector:
    """A mutable 2D vector."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def copy(self) -> "Vector":
        return Vector(self.x, self.y)

    def add(self, other: "Vector") -> None:
        self.x += other.x
        self.y += other.y

    def sub(self, other: "Vector") -> None:
        self.x -= other.x
        self.y -= other.y

    def mult(self, scalar: float) -> None:
        self.x *= scalar
        self.y *= scalar

    def div(self, scalar: float) -> None:
        self.x /= scalar
        self.y /= scalar

    def negate(self) -> None:
        self.x = -self.x
        self.y = -self.y

    def mag(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> None:
        m = self.mag()
        if m != 0:
            self.div(m)

    def set_mag(self, magnitude: float) -> None:
        self.normalize()
        self.mult(magnitude)

    def limit(self, maximum: float) -> None:
        if self.mag() > maximum:
            self.set_mag(maximum)

    def __repr__(self) -> str:
        return f"Vector({self.x!r}, {self.y!r})"


def heading_degrees(vector: Vector) -> float:
    """Direction of a vector in degrees, counterclockwise from +x with y
    pointing down (the convention for orientations)."""
    return -math.atan2(vector.y, vector.x) * (180 / math.pi)


def direction_vector(action_direction: int, directions: int) -> Vector:
    """Unit vector for one of `directions` equally spaced directions;
    direction 0 points to -y."""
    angle = (360 / directions) * action_direction
    radians = math.radians(angle)
    return Vector(math.sin(radians), -math.cos(radians))


class Torus:
    """Distances and offsets on a torus of the given size."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def distance(self, a: Vector, b: Vector) -> float:
        dist_x = min(abs(a.x - b.x), self.width - abs(a.x - b.x))
        dist_y = min(abs(a.y - b.y), self.height - abs(a.y - b.y))
        return math.sqrt(dist_x**2 + dist_y**2)

    def offset(self, origin: Vector, target: Vector) -> tuple[float, float]:
        """Wrapped (dx, dy) from origin to target, each in [-size/2, size/2)."""
        dx = target.x - origin.x
        dy = target.y - origin.y
        dx = (dx + self.width / 2) % self.width - self.width / 2
        dy = (dy + self.height / 2) % self.height - self.height / 2
        return dx, dy

    def away_vector(self, origin: Vector, other: Vector) -> Vector:
        """Offset from origin to other, wrapped by one arena size when a
        component exceeds half of it (the convention for collisions)."""
        dx, dy = other.x - origin.x, other.y - origin.y
        return Vector(
            dx - self.width if abs(dx) > self.width / 2 else dx,
            dy - self.height if abs(dy) > self.height / 2 else dy,
        )

    def in_view(self, observer, position: Vector, view_distance: float, fov: float) -> bool:
        """Whether a position is inside the observer's view cone: within
        view_distance (wrapped) and within fov / 2 of its orientation."""
        dx, dy = self.offset(observer.position, position)
        if math.hypot(dx, dy) > view_distance:
            return False
        angle = -math.degrees(math.atan2(dy, dx))
        diff = abs((angle - observer.orientation + 180) % 360 - 180)
        return diff <= fov / 2
