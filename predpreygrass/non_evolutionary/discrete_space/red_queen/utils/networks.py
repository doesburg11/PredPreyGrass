from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


def build_module_spec(obs_space, act_space, policy_name: str = None):
    """
    Build an RLModuleSpec whose conv depth L matches the observation window size H
    so that the receptive field RF = 1 + 2L equals H.
    Also widens the first FC layer for large action spaces (>20 actions).
    """
    # obs_space is a Box with shape (C, H, W)
    C, H, W = obs_space.shape
    assert H == W and H % 2 == 1, "Expected odd square obs windows (e.g., 7x7, 9x9)."

    # Receptive field math: each 3x3 stride-1 conv layer expands RF by +2.
    # H = 1 + 2L -> L = (H - 1) // 2
    L = (H - 1) // 2

    base_channels = [16, 32, 64]
    if L <= len(base_channels):
        channels = base_channels[:L]
    else:
        channels = base_channels + [64] * (L - len(base_channels))

    conv_filters = [[c, [3, 3], 1] for c in channels]

    num_actions = act_space.n if hasattr(act_space, "n") else None
    if num_actions is not None and num_actions > 20:
        fcnet_hiddens = [384, 256]
        head_note = "wide"
    else:
        fcnet_hiddens = [256, 256]
        head_note = "standard"

    if policy_name is not None:
        rf = 1 + 2 * L
        conv_str = ",".join(str(c) for c in channels)
        print(
            f"[MODEL] {policy_name} -> obs CxHxW={C}x{H}x{W}, "
            f"L={L} (RF={rf}), conv=[{conv_str}], "
            f"actions={num_actions}, head={head_note}"
        )

    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=obs_space,
        action_space=act_space,
        inference_only=False,
        model_config={
            "conv_filters": conv_filters,
            "fcnet_hiddens": fcnet_hiddens,
            "fcnet_activation": "relu",
        },
    )


def build_multi_module_spec(obs_spaces_by_policy: dict, act_spaces_by_policy: dict) -> MultiRLModuleSpec:
    """Build a MultiRLModuleSpec for multiple policies, one RLModuleSpec each."""
    obs_keys = set(obs_spaces_by_policy.keys())
    act_keys = set(act_spaces_by_policy.keys())
    if obs_keys != act_keys:
        missing_in_act = sorted(obs_keys - act_keys)
        missing_in_obs = sorted(act_keys - obs_keys)
        raise ValueError(f"Policy key mismatch. Missing in act: {missing_in_act}; Missing in obs: {missing_in_obs}")

    rl_module_specs = {
        policy_id: build_module_spec(obs_spaces_by_policy[policy_id], act_spaces_by_policy[policy_id], policy_name=policy_id)
        for policy_id in obs_keys
    }
    return MultiRLModuleSpec(rl_module_specs=rl_module_specs)
