## 🔄 Environment Update: Energy and Reproduction Dynamics (v2)

This update introduces a set of critical changes to the predator–prey–grass environment to better reflect **thermodynamic constraints**, **biological realism**, and support for **emergent cooperative behavior**.

The core aim is to move beyond simple reward-driven dynamics by enforcing physical and ecological limitations that shape agent strategy in a more naturalistic way.

---

### 🔧 Key Differences Compared to Earlier Version

| Feature | Before (v1) | Now (v2, this version) | Why It Matters |
|--------|--------------|--------------------------|----------------|
| **Energy intake per event** | Unlimited (agents could gain all energy from a single prey or grass patch) | ✅ **Capped** per prey/grass interaction (configurable) | Models digestion limits and energy conversion inefficiency; encourages sharing and delayed consumption |
| **Absolute energy level** | Unlimited (agents could theoretically accumulate extreme energy levels if not reproducing) | ✅ **Capped** at a species-specific max (configurable) | Reflects biological storage limits; prevents dominance by a few hoarders |
| **Reproduction trigger** | Automatic upon energy threshold | ✅ **Chance-based** (configurable probability) | Introduces variability and risk, encouraging strategic pacing |
| **Reproduction frequency** | No restriction; agents could reproduce every step if above threshold | ✅ **Cooldown between reproductions** (in time steps) | Simulates gestation/recovery time; prevents rapid, unrealistic reproduction loops |
| **Energy loss** | Modeled per step and move (configurable) | ✅ Same, still fully supported | Maintains entropy-producing mechanisms, core to the 2nd Law |
| **Energy transfer efficiency** | 100% efficient transfers | ⚠️ Still 100% in v2 (may be addressed in future) | Real organisms lose energy during transfer — planned for future patch |

---

### 🔬 Why These Changes Matter

- 🧪 **Thermodynamics-Aligned**: No perfect energy storage or infinite reproduction. Energy is bounded, transformed, and partly wasted — consistent with the Second Law.
- 🌱 **Biologically Inspired**: Models limits like stomach capacity, metabolic cost, fertility variation, and reproductive recovery.
- 🤝 **Supports Emergent Cooperation**:
  - Agents cannot hoard forever.
  - Overfeeding yields no benefit — may lead to “sharing” dynamics.
  - Cooldowns and chance introduce strategic gaps exploitable by group timing or avoidance.

---

### 🛠️ New Config Parameters

You can configure the new behaviors using the following fields in `config_env_train.py`:

```python
# Energy intake caps
"max_energy_gain_per_grass": 1.5,
"max_energy_gain_per_prey": 3.5,

# Absolute energy level caps
"max_energy_predator": 20.0,
"max_energy_prey": 14.0,

# Reproduction control
"reproduction_cooldown_steps": 10,
"reproduction_chance_predator": 0.85,
"reproduction_chance_prey": 0.90,
