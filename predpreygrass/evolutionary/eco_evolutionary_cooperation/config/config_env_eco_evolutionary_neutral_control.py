from predpreygrass.evolutionary.eco_evolutionary_cooperation.config.config_env_eco_evolutionary import config_env as _base_config_env

# Neutral-drift control: identical to the real experiment config in every respect
# except genome_neutral_drift_control, so any difference in cooperation_rate drift
# between this run and the real (pilot) run is attributable to selection, not to
# some other config difference. See RESULTS.md.
config_env = {
    **_base_config_env,
    "genome_neutral_drift_control": True,
}
