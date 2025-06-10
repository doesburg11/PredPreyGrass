import pygame
from predpreygrass.pettingzoo.envs.predpreygrass_aec import env
from predpreygrass.pettingzoo.config.config_predpreygrass import env_kwargs

# Initialize pygame
pygame.init()

pygame.display.set_caption("Press SPACE or Click to Continue")

env = env(render_mode="human", **env_kwargs)
env.reset(seed=42)

env.render()
print("Press SPACE or Click to Continue")
# Pause and wait for user input
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Allow closing window
            pygame.quit()
            env.close()
            exit()
        if event.type == pygame.KEYDOWN:  # Wait for spacebar
            if event.key == pygame.K_SPACE:
                waiting = False
        if event.type == pygame.MOUSEBUTTONDOWN:  # Wait for mouse click
            waiting = False

for agent in env.agent_iter():
    print(f"Agent: {agent}")
    observation, reward, termination, truncation, info = env.last()
    if reward > 0.0:
        print(f"Agent: {agent}, Reward: {reward}")

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample()  # Random action

    env.step(action)

    # Pause and wait for user input
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Allow closing window
                pygame.quit()
                env.close()
                exit()
            if event.type == pygame.KEYDOWN:  # Wait for spacebar
                if event.key == pygame.K_SPACE:
                    waiting = False
            if event.type == pygame.MOUSEBUTTONDOWN:  # Wait for mouse click
                waiting = False

env.close()
pygame.quit()
