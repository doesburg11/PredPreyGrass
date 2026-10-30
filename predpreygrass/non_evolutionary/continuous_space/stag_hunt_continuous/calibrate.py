"""Expected-value table of a hunt (DESIGN.md, 3.2), from config_env.

For party sizes 1 to 4, the expected net energy per contributor per hunt
(success chance x share - failure chance x struggle share - join cost),
without bystanders and with one bystander, and per bystander; next to the
energy of a rabbit catch (catch efficiency x rabbit energy at capture) for
a few rabbit energies. The calibration target (DESIGN.md): a pair hunt worth
at least twice a rabbit catch per predator, a solo hunt worth less than
nothing.

    python -m predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.calibrate
"""

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.config.config_env import config_env
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_continuous.environment import PARTY_SIZES


def expected_values(config):
    """[(party size, success chance, per contributor without bystanders,
    per contributor with bystanders, per single bystander)]."""
    carcass = config["hunt_carcass_energy"]
    s = config["hunt_scavenger_fraction"]
    rows = []
    for n in range(1, PARTY_SIZES + 1):
        p = config["hunt_success_by_party"][n - 1]
        failure = (1 - p) * config["hunt_struggle_fraction"] * carcass / n + config["hunt_join_cost"]
        rows.append((n, p, p * carcass / n - failure, p * carcass * (1 - s) / n - failure, p * carcass * s))
    return rows


def main():
    print("party  success  contributor (no bystander)  contributor (bystanders)  one bystander")
    for n, p, alone, shared, bystander in expected_values(config_env):
        print(f"{n:5d}  {p:7.2f}  {alone:26.2f}  {shared:24.2f}  {bystander:13.2f}")
    efficiency = config_env["energy_catch_efficiency_predator"]
    catches = ", ".join(f"{energy:g}: {efficiency * energy:.2f}" for energy in (3, 8, 16))
    print(f"rabbit catch, by rabbit energy at capture: {catches}")


if __name__ == "__main__":
    main()
