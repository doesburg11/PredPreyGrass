from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Genome:
    """Heritable biological traits for one agent.

    The first active traits are deliberately small and interpretable. Policy
    weights are not part of the genome; learned behavior remains within-lifetime
    adaptation unless a future experiment explicitly adds policy inheritance.
    """

    speed: float
    movement_cost_multiplier: float
    reproduction_threshold_multiplier: float
    offspring_energy_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_TRAIT_BOUNDS = {
    "speed": (0.5, 2.0),
    "movement_cost_multiplier": (0.5, 2.0),
    "reproduction_threshold_multiplier": (0.5, 2.0),
    "offspring_energy_fraction": (0.25, 0.9),
}


def _normal_sample(rng: np.random.Generator, mean: float, std: float, bounds: tuple[float, float]) -> float:
    if std <= 0:
        value = mean
    else:
        value = float(rng.normal(mean, std))
    return float(np.clip(value, bounds[0], bounds[1]))


def _bounds(config: Mapping, trait: str) -> tuple[float, float]:
    configured = config.get("trait_bounds", {}).get(trait, DEFAULT_TRAIT_BOUNDS[trait])
    return float(configured[0]), float(configured[1])


def founder_genome(policy_group: str, config: Mapping, rng: np.random.Generator) -> Genome:
    founder_cfg = config.get("founder_genome", {}).get(policy_group, {})
    return Genome(
        speed=_normal_sample(rng, founder_cfg.get("speed_mean", 1.0), founder_cfg.get("speed_std", 0.0), _bounds(config, "speed")),
        movement_cost_multiplier=_normal_sample(
            rng,
            founder_cfg.get("movement_cost_multiplier_mean", 1.0),
            founder_cfg.get("movement_cost_multiplier_std", 0.0),
            _bounds(config, "movement_cost_multiplier"),
        ),
        reproduction_threshold_multiplier=_normal_sample(
            rng,
            founder_cfg.get("reproduction_threshold_multiplier_mean", 1.0),
            founder_cfg.get("reproduction_threshold_multiplier_std", 0.0),
            _bounds(config, "reproduction_threshold_multiplier"),
        ),
        offspring_energy_fraction=_normal_sample(
            rng,
            founder_cfg.get("offspring_energy_fraction_mean", 0.5),
            founder_cfg.get("offspring_energy_fraction_std", 0.0),
            _bounds(config, "offspring_energy_fraction"),
        ),
    )


def mutate_genome(parent: Genome, config: Mapping, rng: np.random.Generator) -> Genome:
    mutation_cfg = config.get("genome_mutation", {})
    rate = float(mutation_cfg.get("rate", 0.0))
    std = float(mutation_cfg.get("std", 0.0))

    values = parent.to_dict()
    for trait, value in list(values.items()):
        if rate > 0 and std > 0 and rng.random() < rate:
            lo, hi = _bounds(config, trait)
            values[trait] = float(np.clip(value + rng.normal(0.0, std), lo, hi))
    return Genome(**values)
