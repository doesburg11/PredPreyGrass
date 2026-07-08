from predpreygrass.eco_evolutionary_metabolic_rate.config.config_env_eco_evolutionary import config_env as _base_config_env

# Neutral-drift control: identical to the real experiment config in every respect
# except genome_neutral_drift_control, so any difference in genome-trait drift
# between this run and a real satiation-throttle run is attributable to selection,
# not to some other config difference. See RESULTS.md.
config_env = {
    **_base_config_env,
    "genome_neutral_drift_control": True,
}
