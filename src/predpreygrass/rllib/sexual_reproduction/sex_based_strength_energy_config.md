# Sex-based Physical Strength Abstraction (Energy Economics)

This document defines a **biology-inspired, RL-friendly abstraction** for modeling
average physical differences between males and females **without hard-coded dominance**.

The parameters are intended for use in **PredPreyGrass**-style environments, where
differences emerge through **energy budgets, costs, and risks** rather than fixed traits.

---

## Design principles

- No direct `strength` parameter
- Differences expressed via:
  - Initial energy
  - Maintenance costs
  - Action costs
  - Risk penalties
  - Energy extraction efficiency
- Large overlap between sexes is preserved
- PPO / MARL can exploit or ignore differences freely

---

## Copy-ready configuration block

```python
# ============================================================
# Sex-based physical strength abstraction (energy economics)
# ============================================================

# --------------------
# Initial energy
# --------------------
E_INIT = {
    "male": 120,
    "female": 100,
}

# --------------------
# Baseline maintenance cost (idle metabolism)
# --------------------
IDLE_COST = {
    "male": 1.1,
    "female": 1.0,
}

# --------------------
# Action energy costs
# --------------------
ACTION_COST = {
    "attack": {
        "male": 8,
        "female": 10,
    },
    "move": {
        "male": 2.2,
        "female": 2.0,
    },
}

# --------------------
# Energy extraction efficiency from kills
# --------------------
KILL_EFFICIENCY = {
    "male": 1.1,
    "female": 1.0,
}

# --------------------
# Risk / failure penalties
# --------------------
FAILED_ATTACK_DAMAGE = {
    "male": 15,
    "female": 10,
}

# --------------------
# (Optional) Reproduction energy cost placeholder
# --------------------
REPRO_COST = {
    "male": 40,
    "female": 100,
}
```

---

## Interpretation notes

- **Males**
  - Higher burst capacity
  - Higher idle burn
  - Higher risk cost when failing
- **Females**
  - Slightly lower peak energy
  - Better efficiency in sustained actions
  - Lower failure penalties

No behavioral assumptions are imposed — all differentiation must be *learned*.

---

## Intended use

- Energy initialization
- Cost calculation per step
- Attack / cooperation dynamics
- Sexual reproduction extensions
- Stag-hunt–like coordination experiments

This setup is deliberately conservative to avoid baked-in outcomes.
