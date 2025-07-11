## 🔄 Environment Update: Energy and Reproduction Dynamics (v2.1)

This update introduces a set of critical changes to the predator–prey–grass environment to better reflect **thermodynamic constraints**, **biological realism**, and support for **emergent cooperative behavior**.

The core aim is to move beyond simple reward-driven dynamics by enforcing physical and ecological limitations that shape agent strategy in a more naturalistic way.

---

### 🔧 Key Differences Compared to Earlier Version

| Feature | Before (v1) | Now (v2.1, this version) | Why It Matters |
|--------|--------------|---------------------------|----------------|
| **Energy intake per event** | Unlimited (agents could gain all energy from a single prey or grass patch) | ✅ **Capped** per prey/grass interaction (configurable) | Models digestion limits and energy conversion inefficiency; encourages sharing and delayed consumption |
| **Absolute energy level** | Unlimited (agents could theoretically accumulate extreme energy levels if not reproducing) | ✅ **Capped** at a species-specific max (configurable) | Reflects biological storage limits; prevents dominance by a few hoarders |
| **Reproduction trigger** | Automatic upon energy threshold | ✅ **Chance-based** (configurable probability) | Introduces variability and risk, encouraging strategic pacing |
| **Reproduction frequency** | No restriction; agents could reproduce every step if above threshold | ✅ **Cooldown between reproductions** (in time steps) | Simulates gestation/recovery time; prevents rapid, unrealistic reproduction loops |
| **Energy loss per step/move** | Configurable | ✅ Still supported | Maintains entropy-producing behavior and energy cost |
| **Energy transfer efficiency (eating)** | 100% efficient (all energy from grass/prey is gained) | ✅ **Configurable η < 1.0** (e.g., 0.85) | Models digestive/metabolic loss; aligns with Second Law; encourages more frequent foraging |
| **Reproduction energy efficiency** | 100% transfer (all parental energy passed to offspring) | ✅ **Configurable η < 1.0** (e.g., 0.85) | Models biological inefficiency in reproduction; reduces runaway population loops |

---

### 🔬 Why These Changes Matter

- 🧪 **Thermodynamic Consistency**: Energy is no longer perfectly conserved during transfer — matching the 2nd Law. All biological processes now involve energy dissipation.
- 🌱 **Ecological Realism**: Energy transfer caps and losses match real-world biology (digestion, storage limits, reproduction cost).
- 🤝 **Supports Emergent Social Behavior**:
  - Overfeeding is discouraged.
  - Reproduction becomes a meaningful investment.
  - Resource turnover dynamics allow sharing, patience, and differentiation.

---

### 🛠️ New Config Parameters

You can configure the updated energy and reproduction rules with the following fields in `config_env_train.py` and `config_env_eval.py`:

```python
# Energy intake caps
"max_energy_gain_per_grass": 1.5,
"max_energy_gain_per_prey": 3.5,

# Absolute energy caps
"max_energy_predator": 20.0,
"max_energy_prey": 14.0,

# Reproduction control
"reproduction_cooldown_steps": 10,
"reproduction_chance_predator": 0.85,
"reproduction_chance_prey": 0.90,

# Energy transfer efficiency (digestion loss)
"energy_transfer_efficiency": 0.85,

# Reproduction efficiency (offspring receives η × parent energy investment)
"reproduction_energy_efficiency": 0.85,
