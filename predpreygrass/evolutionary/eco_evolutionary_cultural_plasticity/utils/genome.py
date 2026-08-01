from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Genome:
    """Heritable dual-inheritance genome for one agent: gene-culture coevolution.

    Two channels, not one:
    - `plasticity` (continuous, [0, 1]): the genuinely Darwinian trait. It sets
      the per-check probability that this agent's *live* dialect (see
      predpreygrass_rllib_env.py's `_apply_cultural_learning`) updates toward
      the locally observed majority dialect. Genes here do not encode behavior
      directly -- they encode capacity to adopt culture, the literal
      Baldwin/dual-inheritance mechanism (Boyd & Richerson).
    - `dialect` (categorical, [0, n_dialects)): the agent's *founder* cultural
      bias, inherited like any other genome field but immediately overridable
      within the agent's own lifetime by `_apply_cultural_learning`. The live,
      mutable value lives in `env.agent_live_dialect`, not here -- this field
      is only the starting point a newborn is seeded with.

    Policy weights are not part of the genome; movement/hunting behavior
    remains within-lifetime PPO adaptation, shared per species as in every
    prior eco_evolutionary module.
    """

    plasticity: float
    dialect: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_TRAIT_BOUNDS = {
    "plasticity": (0.0, 1.0),
}

GENOME_FIELD_DEFAULTS: dict[str, float] = {
    "plasticity": 0.1,
    "dialect": 0,
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


def _n_dialects(config: Mapping) -> int:
    return int(config.get("n_dialects", 4))


def founder_genome(policy_group: str, config: Mapping, rng: np.random.Generator) -> Genome:
    founder_cfg = config.get("founder_genome", {}).get(policy_group, {})
    plasticity = _normal_sample(
        rng,
        founder_cfg.get("plasticity_mean", 0.1),
        founder_cfg.get("plasticity_std", 0.05),
        _bounds(config, "plasticity"),
    )
    dialect = int(rng.integers(_n_dialects(config)))
    return Genome(plasticity=plasticity, dialect=dialect)


def mutate_genome(parent: Genome, config: Mapping, rng: np.random.Generator) -> Genome:
    mutation_cfg = config.get("genome_mutation", {})
    rate = float(mutation_cfg.get("rate", 0.0))
    std = float(mutation_cfg.get("std", 0.0))

    plasticity = parent.plasticity
    if rate > 0 and std > 0 and rng.random() < rate:
        lo, hi = _bounds(config, "plasticity")
        plasticity = float(np.clip(plasticity + rng.normal(0.0, std), lo, hi))

    dialect_mutation_cfg = config.get("dialect_mutation", {})
    dialect_rate = float(dialect_mutation_cfg.get("rate", 0.0))
    dialect = parent.dialect
    if dialect_rate > 0 and rng.random() < dialect_rate:
        # Categorical mutation: resample uniformly among all dialects (may land
        # back on the same one), same convention as the combinatorial loci
        # genome's per-locus mutation in eco_evolutionary_metabolic_code.
        dialect = int(rng.integers(_n_dialects(config)))

    return Genome(plasticity=plasticity, dialect=dialect)
