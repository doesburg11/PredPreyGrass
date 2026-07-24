from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

# Locus states, defined relative to an implicit fixed target (Hinton & Nowlan, 1987):
# only relative state matters, so no target bit-string needs to be stored anywhere.
CORRECT = 1   # permanently matches the target -- no action needed
WRONG = -1    # permanently mismatches the target -- unfixable within this lifetime
PLASTIC = 0   # unresolved -- re-guessed every step until solved (see attempt_resolve)

DEFAULT_NUM_LOCI = 10

DEFAULT_FOUNDER_PROBS = {
    "correct": 0.2,
    "wrong": 0.3,
    "plastic": 0.5,
}

_STATE_BY_LABEL = {"correct": CORRECT, "wrong": WRONG, "plastic": PLASTIC}


@dataclass(frozen=True)
class Genome:
    """Heritable trait for one agent: a combinatorial locus code.

    Policy weights are not part of the genome; learned behavior remains
    within-lifetime PPO adaptation. This genome instead drives a *separate*,
    genuinely per-individual lifetime search (see predpreygrass_rllib_env.py's
    attempt_resolve loop) -- a second, independent "learning" channel modeled
    directly on Hinton & Nowlan (1987) rather than approximated through PPO.

    loci: length-L tuple of CORRECT/WRONG/PLASTIC states. An agent can only
    ever achieve a full match if it carries zero WRONG loci; PLASTIC loci are
    searched fresh every step within that agent's own lifetime.
    """

    loci: tuple[int, ...]

    def to_dict(self) -> dict[str, tuple[int, ...]]:
        return asdict(self)

    @property
    def num_wrong(self) -> int:
        return sum(1 for locus in self.loci if locus == WRONG)

    @property
    def num_correct(self) -> int:
        return sum(1 for locus in self.loci if locus == CORRECT)

    @property
    def num_plastic(self) -> int:
        return sum(1 for locus in self.loci if locus == PLASTIC)

    @property
    def has_wrong(self) -> bool:
        return self.num_wrong > 0


GENOME_FIELD_DEFAULTS: dict[str, tuple[int, ...]] = {
    "loci": tuple([PLASTIC] * DEFAULT_NUM_LOCI),
}


def _num_loci(config: Mapping) -> int:
    return int(config.get("haystack_num_loci", DEFAULT_NUM_LOCI))


def _founder_probs(config: Mapping, policy_group: str) -> dict[str, float]:
    per_species = config.get("haystack_founder_probs", {}).get(policy_group, {})
    return {
        "correct": float(per_species.get("correct", DEFAULT_FOUNDER_PROBS["correct"])),
        "wrong": float(per_species.get("wrong", DEFAULT_FOUNDER_PROBS["wrong"])),
        "plastic": float(per_species.get("plastic", DEFAULT_FOUNDER_PROBS["plastic"])),
    }


def _sample_locus(rng: np.random.Generator, probs: dict[str, float]) -> int:
    labels = ("correct", "wrong", "plastic")
    p = np.array([probs[label] for label in labels], dtype=np.float64)
    p = p / p.sum()
    choice = rng.choice(3, p=p)
    return _STATE_BY_LABEL[labels[int(choice)]]


def founder_genome(policy_group: str, config: Mapping, rng: np.random.Generator) -> Genome:
    probs = _founder_probs(config, policy_group)
    num_loci = _num_loci(config)
    loci = tuple(_sample_locus(rng, probs) for _ in range(num_loci))
    return Genome(loci=loci)


def mutate_genome(parent: Genome, config: Mapping, rng: np.random.Generator) -> Genome:
    mutation_cfg = config.get("haystack_mutation", {})
    rate = float(mutation_cfg.get("rate", 0.0))

    states = (CORRECT, WRONG, PLASTIC)
    new_loci = list(parent.loci)
    if rate > 0:
        for i, locus in enumerate(new_loci):
            if rng.random() < rate:
                new_loci[i] = int(states[int(rng.integers(3))])
    return Genome(loci=tuple(new_loci))


def attempt_resolve(loci: tuple[int, ...], rng: np.random.Generator) -> bool:
    """One lifetime-search trial: a fresh joint guess across all PLASTIC loci.

    Mirrors Hinton & Nowlan's mechanism -- the whole combination is tested at
    once, not locus-by-locus, so an individual with k plastic loci needs on
    average 2**k trials. Callers should skip agents with any WRONG locus
    entirely (they can never succeed) rather than calling this for them.
    """
    for locus in loci:
        if locus == WRONG:
            return False
        if locus == PLASTIC and rng.random() >= 0.5:
            return False
    return True
