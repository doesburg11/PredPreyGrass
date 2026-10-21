"""Small PyTorch DQN and DDPG implementations for the SELFish experiment."""

from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, hidden_layers: int = 10, width: int = 16):
        super().__init__()
        layers: list[nn.Module] = []
        size = observation_size
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(size, width), nn.ReLU()))
            size = width
        layers.append(nn.Linear(size, 5))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations):
        return self.layers(observations)


class Actor(nn.Module):
    def __init__(self, observation_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        size = observation_size
        for _ in range(5):
            layers.extend((nn.Linear(size, 16), nn.ReLU()))
            size = 16
        layers.extend((nn.Linear(size, 1), nn.Tanh()))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations):
        return self.layers(observations)


class Critic(nn.Module):
    def __init__(self, observation_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        size = observation_size + 1
        for _ in range(5):
            layers.extend((nn.Linear(size, 32), nn.ReLU()))
            size = 32
        layers.append(nn.Linear(size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, observations, actions):
        return self.layers(torch.cat((observations, actions), dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.items = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, done):
        self.items.append(
            (
                np.asarray(observation, dtype=np.float32).copy(),
                float(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                float(done),
            )
        )

    def sample(self, batch_size: int, device):
        indices = np.random.randint(len(self.items), size=batch_size)
        batch = [self.items[index] for index in indices]
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return (
            torch.as_tensor(np.stack(observations), device=device),
            torch.as_tensor(np.asarray(actions, dtype=np.float32).reshape(-1, 1), device=device),
            torch.as_tensor(np.asarray(rewards, dtype=np.float32).reshape(-1, 1), device=device),
            torch.as_tensor(np.stack(next_observations), device=device),
            torch.as_tensor(np.asarray(dones, dtype=np.float32).reshape(-1, 1), device=device),
        )

    def __len__(self):
        return len(self.items)


def frozen_copy(module):
    snapshot = copy.deepcopy(module).eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def dqn_update(online, target, optimizer, replay, batch_size, gamma, device):
    observations, actions, rewards, next_observations, dones = replay.sample(batch_size, device)
    actions = actions.long()
    predicted = online(observations).gather(1, actions)
    with torch.no_grad():
        expected = rewards + gamma * (1.0 - dones) * target(next_observations).max(dim=1, keepdim=True).values
    loss = nn.functional.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def ddpg_update(
    actor,
    actor_target,
    critic,
    critic_target,
    actor_optimizer,
    critic_optimizer,
    replay,
    batch_size,
    gamma,
    tau,
    device,
):
    observations, actions, rewards, next_observations, dones = replay.sample(batch_size, device)
    with torch.no_grad():
        next_actions = actor_target(next_observations)
        targets = rewards + gamma * (1.0 - dones) * critic_target(next_observations, next_actions)
    critic_loss = nn.functional.mse_loss(critic(observations, actions), targets)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = -critic(observations, actor(observations)).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()

    with torch.no_grad():
        for online, target in ((actor, actor_target), (critic, critic_target)):
            for source, destination in zip(online.parameters(), target.parameters()):
                destination.mul_(1.0 - tau).add_(source, alpha=tau)
    return float(critic_loss.detach()), float(actor_loss.detach())
