# stag_hunt_rule_prey: results

Calibration only, one seed per arm, no confirmatory test (see the [README](README.md)).

## 1. Prey on their own

One harmless hunter (it stands still, catches and frightens nothing, and has no
young), default settings, 12 episodes (`results/prey_check_default.json`):

| From step | Rabbits | Mammoths | Mammoth energy | Edible grass | Edible steppe |
|---|---|---|---|---|---|
| 0 | 50 | 16 | 13 | 12 | 5.4 |
| 1000 | 81 | 20 | 15 | 4.7 | 4.0 |
| 3000 | 83 | 20 | 21 | 5.3 | 3.7 |
| 7000 | 83 | 20 | 25 | 4.7 | 3.3 |

No extinctions. Rabbits settle near 83, under their cap of 100, so grass limits them.
Mammoths stay at their cap of 20, where they cannot breed, and keep fattening, so
unhunted mammoths drift out of a party's reach: at ratio 0.6 a mammoth of 25 needs
a party above 15 energy, three typical hunters. Hunting should lower this by making
room for births, not measured yet.

Training, evaluation, accounts and the viewer have been run on a two-iteration
smoke run. All tests pass.

## 2. Prey at other food levels

Same check, 12 episodes per level, not committed as result files. The ordinary
regrowth delays (`*_respawn_delay`, 200 and 800) changed almost nothing, because
clusters get grazed out and a collapse overwrites that delay. The delay after a
collapse (`*_overgrazing_delay`) is what matters:

| Grass collapse delay | Rabbits |
|---|---|
| 200 | 100 (at the cap) |
| 400 (default) | 83 |
| 800 | 46 |
| 1600 | dying out, gone in 7 of 12 |

| Steppe collapse delay | Mammoths | Mammoth energy |
|---|---|---|
| 400 (default) | 20 (at the cap) | rising to 25 |
| 800 | 15 | about 10 |
| 1600 | 7-9 | 6-9 |
| 3200 | gone in all 12 | |

## 3. Calibration (2026-11-05, running)

Arms `base` (collapse delays 400) and `lean` (grass and steppe 800), 1000 iterations,
started 09:22. Results follow here when both finish; this ends the calibration
whatever the result.
