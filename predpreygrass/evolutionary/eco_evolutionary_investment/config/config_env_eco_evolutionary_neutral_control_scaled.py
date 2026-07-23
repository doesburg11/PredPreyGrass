from predpreygrass.evolutionary.eco_evolutionary_investment.config.config_env_eco_evolutionary_scaled import config_env as _base_config_env

# Trial 6 (population scaling) neutral-drift control: identical to
# config_env_eco_evolutionary_scaled.py (2x grid/population) in every respect
# except genome_neutral_drift_control, so any difference in
# offspring_investment_fraction drift between this run and the Trial 6 real run
# (seed 42, 2026-07-18/19) is attributable to selection, not some other config
# difference. See predpreygrass/evolutionary/RESULTS.md, Trial 6.
config_env = {
    **_base_config_env,
    "genome_neutral_drift_control": True,
}
