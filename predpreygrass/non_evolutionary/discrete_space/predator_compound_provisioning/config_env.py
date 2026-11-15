"""Configuration for Iteration 20: compound movement-plus-provisioning actions."""
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.config_env import config_env as BASE_CONFIG


config_env = {
    **BASE_CONFIG,
    # 18 predator actions: 0..8 move without giving; 9..17 make the same
    # respective move while requesting a gift after movement.  Unlike
    # Iteration 19's action 9, a transfer is never a duplicate no-op.
    "compound_gift": "on",  # "on" transfers; "inert" is the matched 18-action control
    "learned_gift": False,  # must remain off: this module owns the compound action encoding
    "learned_gift_amount": 2.0,
    "male_gift_donation_rate": 0.0,  # no automatic gift
    "mate_obs_channel": True,
}
