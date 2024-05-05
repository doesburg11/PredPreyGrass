import pygame
import random

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Define the size of the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Define the size of each graph
GRAPH_WIDTH = 200
GRAPH_HEIGHT = 150

# Define a class for the graph
class Graph:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data = [random.randint(0, 100) for _ in range(50)]  # Example data for the graph

    def draw(self, screen):
        # Draw the graph background
        pygame.draw.rect(screen, WHITE, (self.x, self.y, GRAPH_WIDTH, GRAPH_HEIGHT))

        # Draw the graph lines (example)
        for i in range(len(self.data) - 1):
            pygame.draw.line(screen, BLACK, (self.x + i * 4, self.y + GRAPH_HEIGHT - self.data[i]),
                             (self.x + (i + 1) * 4, self.y + GRAPH_HEIGHT - self.data[i + 1]), 2)


def main():
    pygame.init()

    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Multiple Graphs Example")

    # Create graph objects
    graph1 = Graph(50, 50)
    graph2 = Graph(300, 50)

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear the screen
        screen.fill(BLACK)

        # Draw the graphs
        graph1.draw(screen)
        graph2.draw(screen)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
