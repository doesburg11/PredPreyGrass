# Mammoths (big game that needs a group): design draft (not built; 2026-09-26)

Status (2026-09-26): BUILT as a default-off option of `predator_bands` (like the compass, threats and distance decay), Codex-reviewed, 194
module tests. Differences from this draft: a tie between parties favours the attacker (not random); a hunt is resolved where the mammoth
stands before it wanders; mammoths require bands (`num_bands > 0`); a mammoth share counts as eating for `reward_predator_step`.
First pilot (3 bands, 2 mammoths, seed 42, 150 iterations): the population is viable but NO joint hunt was learned; see the README.

## Why

Every attempt to make bands cohere with a small payoff failed (compass, threats, wider defense, distance-scaled sharing; see the
README): predators keep moving away from any neighbour because neighbours compete for food and cells. A payoff for being together has
to beat that competition. Real bands hunt big game together and share the kill. A mammoth is a resource that only a group can take,
worth far more than a single prey. No rule tells anyone where to stand and no reward term is added: the energy is the payoff.

## Mechanics

- **Entity.** `num_mammoths` (default 0 = off) non-learning animals with `mammoth_energy` (default 20; ordinary prey is about 3).
  They live outside `agents` like the scripted prey and threats, on their own position dict. They wander slowly (move with probability
  `mammoth_move_prob` = 0.3 per step) and never flee. A killed mammoth is replaced after `mammoth_respawn_steps` (default 80) at a random
  free cell with full energy, so big game is a renewable resource.
- **Attack (a mammoth hunt is a band hunt; decided 2026-09-26).** A predator that steps onto a mammoth's cell starts an attempt (the same
  trigger as hunting ordinary prey; predators never share a cell with each other but may share one with prey or a mammoth). The party is
  the attacker's band-mates that are viable and within Chebyshev distance 1 of the mammoth (`mammoth_party_radius`), plus the attacker.
  Predators of other bands do not count: they add nothing to the odds, get no share and cannot die in the attempt. Ethnographically, big-game
  parties are mostly drawn from one camp, and cooperation between camps is the exception. One attempt is resolved per mammoth per step: if
  members of several bands are on the mammoth's cell or next to it, the band with the larger party acts (ties at random) and the others
  are left out that step.
- **Outcome by party size n** (`mammoth_success_by_party`, `mammoth_death_by_party`, indexed n = 1, 2, 3, 4+; first guesses):

  | n | success | each party member dies on failure |
  |---|---|---|
  | 1 | 0.02 | 0.30 |
  | 2 | 0.25 | 0.15 |
  | 3 | 0.60 | 0.05 |
  | 4+ | 0.90 | 0.02 |

  A lone hunter almost always fails and is likely killed; a party of four almost always succeeds. This is the group-size dependence you
  proposed, in the environment's physics.
- **Sharing the kill.** The mammoth's energy is split equally among the party (the band-mates who took part). It is meat
  (raises total energy and the meat store, not the fruit store). Each recipient also gets the ordinary energy-proportional reward for
  the energy it received (`reward_predator_per_energy`), the same rule as for any forage, so cooperation is paid in the currency
  everything else is paid in. Not needed with the single-band party rule; a band contest with an exponent r
  would only matter if cross-band hunts were allowed.
- **Ordinary prey stay.** Solo hunting and gathering still work, so grouping has to beat going alone, not replace starving.

## Observation

One new channel marks mammoths in the predator's window, valued by energy (so the policy can tell it from small prey); it goes after the
compass and threat channels. Party size is already visible through the predator layer (neighbours' positions and energies).

## What is learned, and the hard part

Learned: whether to go to a mammoth, when to attack, and whether to wait for company. Hard: the attack is a rare joint event early on
(three or four predators must arrive together), so independent PPO learners may never discover it. The success curve gives a weak but
non-zero signal at n = 1 and 2; if it is still not found in a pilot, the fixes are a smoother curve (success 0.1 at n = 1) or the band
compass so members can converge on a mammoth. The band compass and mammoths are complementary and I would test them together after the
plain version.

## Calibration and metrics

Abundance, not danger, is the first knob: `num_mammoths` (2) and `mammoth_respawn_steps` (80) set how much big-game meat there is; with 3
bands (18 predators) each mammoth kill feeds a party of about 4 with about 5 energy each. Metrics: attempts and kills by party size,
deaths on failure by party size, energy distributed, party size distribution, attempts blocked because a larger party of another band was present.
Analysis: the scatter analysis (does out-of-sharing-range drop against the control?), and a new one: at attempts, how close are party
members to their band-mates before the attempt.

## Decisions for you

1. **Attack trigger:** stepping onto the mammoth's cell (as with prey), or being adjacent? I propose stepping on: it reuses the hunting
   pattern and lets one predator commit while others arrive.
2. ~~Split rule and cross-band hunts~~ **Decided:** single-band parties, equal split within the party (see Attack above).
3. **Death on failure:** each party member with the probabilities above (dangerous, realistic), or no deaths (only wasted effort)?
4. **Respawn:** 80 steps at a random cell (renewable), or a fixed herd that is depleted?

## Build plan

Tests first (party-size arithmetic with forced rolls, equal split and reward, respawn, observation channel, snapshot, default-off equals
the current behaviour), then a Codex review, then a pilot: 3 bands, 2 mammoths, no threats, 150 iterations, against the no-threat control,
with the scatter analysis and the party-size metrics.
