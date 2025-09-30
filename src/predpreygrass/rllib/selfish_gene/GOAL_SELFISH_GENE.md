# Goal: Tier-1 Selfish Gene Reward
Principle: reward = lineage replication only (no Hamilton/Price, no scripted social verbs).
Minimal changes:
- Each agent carries: lineage_tag (inherited), parent_id, birth_step.
- Window W (e.g., 2000): per-agent direct fitness = #viable offspring in W (option: weight by offspring survival at window end).
- Reward: R_i^(W) = α * (direct_fitness_i / mean_births_per_capita_W). Trickle evenly across the last W steps to reduce variance.
Guards: cap births/parent/short-horizon; discount closely spaced births.
Keep base ecology (eat, starve, die) but reduce stepwise shaping so Tier-1 dominates.
