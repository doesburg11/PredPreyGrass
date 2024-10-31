import pygame
import random

# Define constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
CELL_SIZE = WIDTH // GRID_SIZE
GRASS_REGROWTH_TIME = 10  # Steps required for grass to regrow

# Colors
GREEN = (0, 255, 0)    # Grass
RED = (255, 0, 0)      # Predator
BLUE = (0, 0, 255)     # Prey
BROWN = (139, 69, 19)   # Empty cell

class Grass:
    def __init__(self):
        self.alive = True
        self.age = 0  # Age of the grass for regrowth tracking

    def eaten(self):
        self.alive = False
        self.age = 0  # Reset age when eaten

    def regrow(self):
        if not self.alive:
            self.age += 1
            if self.age >= GRASS_REGROWTH_TIME:
                self.alive = True  # Regrow grass after reaching the regrowth time
                self.age = 0

class Prey:
    def __init__(self):
        self.alive = True
        self.energy = 50  # Initial energy

    def move(self, grid):
        if not self.alive:
            return

        # Randomly choose a direction to move
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # down, right, up, left
        dx, dy = random.choice(directions)
        new_x = (self.x + dx) % GRID_SIZE
        new_y = (self.y + dy) % GRID_SIZE

        # Check if the new cell contains grass
        if isinstance(grid.get_cell(new_x, new_y), Grass):
            grass = grid.get_cell(new_x, new_y)
            if grass.alive:
                # Eat the grass and gain energy
                self.energy += 10  # Gain energy from eating grass
                grass.eaten()      # Mark grass as eaten
                grid.grid[self.x][self.y] = None  # Remove prey from its old position
                grid.grid[new_x][new_y] = self  # Move prey to new position
                self.x, self.y = new_x, new_y
        elif grid.get_cell(new_x, new_y) is None:
            # Move the prey to the new cell if it's empty
            grid.grid[new_x][new_y] = self
            grid.grid[self.x][self.y] = None
            self.x, self.y = new_x, new_y

        # Decrease energy
        self.energy -= 1
        if self.energy <= 0:
            self.alive = False
            grid.grid[self.x][self.y] = None  # Remove from grid when dead

class Predator:
    def __init__(self):
        self.alive = True
        self.energy = 50  # Initial energy

    def move(self, grid):
        if not self.alive:
            return

        # Randomly choose a direction to move
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # down, right, up, left
        dx, dy = random.choice(directions)
        new_x = (self.x + dx) % GRID_SIZE
        new_y = (self.y + dy) % GRID_SIZE

        # Check if the new cell contains prey
        if isinstance(grid.get_cell(new_x, new_y), Prey):
            prey = grid.get_cell(new_x, new_y)
            # Eat the prey and transfer energy
            self.energy += prey.energy  # Transfer energy
            prey.alive = False            # The prey is now dead
            grid.grid[self.x][self.y] = None  # Remove predator from its old position
            grid.grid[new_x][new_y] = self  # Move predator to new position
            self.x, self.y = new_x, new_y
        elif grid.get_cell(new_x, new_y) is None:
            # Move the predator to the new cell if it's empty
            grid.grid[new_x][new_y] = self
            grid.grid[self.x][self.y] = None
            self.x, self.y = new_x, new_y

        # Decrease energy
        self.energy -= 1
        if self.energy <= 0:
            self.alive = False
            grid.grid[self.x][self.y] = None  # Remove from grid when dead

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.populate_grid()

    def populate_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                rand_num = random.random()
                if rand_num < 0.1:
                    grass = Grass()
                    self.grid[i][j] = grass
                elif rand_num < 0.2:
                    prey = Prey()
                    prey.x, prey.y = i, j
                    self.grid[i][j] = prey
                elif rand_num < 0.3:
                    predator = Predator()
                    predator.x, predator.y = i, j
                    self.grid[i][j] = predator

    def get_cell(self, x, y):
        return self.grid[x][y]

    def update_grass(self):
        # Regrow grass in the grid
        for i in range(self.size):
            for j in range(self.size):
                entity = self.get_cell(i, j)
                if isinstance(entity, Grass):
                    entity.regrow()

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.grid = Grid(GRID_SIZE)

    def render(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                entity = self.grid.get_cell(i, j)
                if isinstance(entity, Grass) and entity.alive:
                    color = GREEN
                elif isinstance(entity, Prey):
                    color = BLUE
                elif isinstance(entity, Predator):
                    color = RED
                else:
                    color = BROWN  # Empty cell

                pygame.draw.rect(self.screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def update(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                entity = self.grid.get_cell(i, j)
                if isinstance(entity, Prey):
                    entity.move(self.grid)
                elif isinstance(entity, Predator):
                    entity.move(self.grid)
        self.grid.update_grass()  # Update grass regrowth after all movements

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(BROWN)  # Fill background
            self.update()             # Update movements and grass
            self.render()             # Render the grid
            pygame.display.flip()     # Update the display
            self.clock.tick(10)       # Control the frame rate (10 FPS)

        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()
