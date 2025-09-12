
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

# Ensure custom models are registered on import
try:  # noqa: F401
    import predpreygrass.rllib.dynamic_field_theory.models.dft_memory_model  # registers DFTMemoryConvModel
except Exception:
    pass


def build_module_spec(obs_space, act_space, policy_name: str = None):
    """
    Build an RLModuleSpec whose conv depth L matches the observation window size H
    so that the receptive field RF = 1 + 2L equals H.
    Also widens the first FC layer for large action spaces (>20 actions).
    """   
    # obs_space is a Box with shape (C, H, W)
    #   C = number of channels (layers of information: mask, predators, prey, grass → usually 4)
    #   H = height of the square observation window (e.g. 7 for predators, 9 for prey)
    #   W = width of the square observation window (equal to H here)
    C, H, W = obs_space.shape

    # We assert the window is square and odd-sized.
    # Odd size is important because the agent is always centered in its window.
    assert H == W and H % 2 == 1, "Expected odd square obs windows (e.g., 7x7, 9x9)."

    # Receptive field math:
    # Each 3x3 stride-1 conv layer expands the receptive field by +2.
    # General formula: RF = 1 + 2 * L
    # To cover the full observation window (size H), we solve:
    #   H = 1 + 2L  →  L = (H - 1) // 2
    # Example: H=7 → L=3 conv layers (RF=7), H=9 → L=4 conv layers (RF=9).
    L = (H - 1) // 2

    # Channel schedule:
    # We start small (16 filters), then increase (32, 64).
    # If more layers are needed (e.g., prey with 9x9), we keep adding 64-channel layers.
    base_channels = [16, 32, 64]
    if L <= len(base_channels):
        channels = base_channels[:L]
    else:
        channels = base_channels + [64] * (L - len(base_channels))

    # Assemble conv_filters list for RLlib (format: [num_filters, [kernel, kernel], stride])
    conv_filters = [[c, [3, 3], 1] for c in channels]

    # Adjust the fully-connected (FC) hidden sizes based on the action space.
    # If the agent has many actions (e.g., prey with 25 moves), we give it a wider first FC layer
    # so it has more capacity to rank actions effectively.
    num_actions = act_space.n if hasattr(act_space, "n") else None
    if num_actions is not None and num_actions > 20:
        fcnet_hiddens = [384, 256]
        head_note = "wide"
    else:
        fcnet_hiddens = [256, 256]
        head_note = "standard"

    # ---- Debug/trace log (once per policy) ----
    # Example: [MODEL] type_1_prey → obs CxHxW=4x9x9, L=4 (RF=9), conv=[16,32,64,64], actions=25, head=wide
    if policy_name is not None:
        rf = 1 + 2 * L
        conv_str = ",".join(str(c) for c in channels)
        print(
            f"[MODEL] {policy_name} → obs CxHxW={C}x{H}x{W}, "
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

def build_multi_module_spec(
    obs_spaces_by_policy: dict,
    act_spaces_by_policy: dict,
) -> MultiRLModuleSpec:
    """
    Build a MultiRLModuleSpec for multiple policies.

    Args:
        obs_spaces_by_policy: dict mapping policy_id -> observation_space
        act_spaces_by_policy: dict mapping policy_id -> action_space

    Returns:
        MultiRLModuleSpec containing an RLModuleSpec per policy, using
        DefaultPPOTorchRLModule with conv/FC settings chosen by build_module_spec().

    Notes:
        - Keys of obs_spaces_by_policy and act_spaces_by_policy must match.
        - This function prints one model summary line per policy (via build_module_spec).
    """
    # Ensure both dicts have the same policy IDs
    obs_keys = set(obs_spaces_by_policy.keys())
    act_keys = set(act_spaces_by_policy.keys())
    if obs_keys != act_keys:
        missing_in_act = sorted(obs_keys - act_keys)
        missing_in_obs = sorted(act_keys - obs_keys)
        raise ValueError(
            f"Policy key mismatch. "
            f"Missing in act: {missing_in_act}; Missing in obs: {missing_in_obs}"
        )

    rl_module_specs = {}
    for policy_id in obs_keys:
        rl_module_specs[policy_id] = build_module_spec(
            obs_spaces_by_policy[policy_id],
            act_spaces_by_policy[policy_id],
            policy_name=policy_id,  
        )

    return MultiRLModuleSpec(rl_module_specs=rl_module_specs)


def build_multi_module_spec_agent_dft(
    obs_spaces_by_policy: dict,
    act_spaces_by_policy: dict,
    *,
    prey_channel_index: int = 2,
    dft_params: dict | None = None,
) -> MultiRLModuleSpec:
    """Build a MultiRLModuleSpec using the custom DFTMemoryConvModel.

    Notes:
        - Observation spaces are unchanged; the model augments with a memory channel internally.
        - Provide dft_params to override defaults: keys like dft_dt, dft_tau, dft_gamma,
          dft_exc_sigma, dft_exc_gain, dft_input_gain, dft_zmin, dft_zmax, dft_kernel_size.
    """
    obs_keys = set(obs_spaces_by_policy.keys())
    act_keys = set(act_spaces_by_policy.keys())
    if obs_keys != act_keys:
        missing_in_act = sorted(obs_keys - act_keys)
        missing_in_obs = sorted(act_keys - obs_keys)
        raise ValueError(
            f"Policy key mismatch. Missing in act: {missing_in_act}; Missing in obs: {missing_in_obs}"
        )

    # Base config shared across policies; channel will be set per-policy below
    custom_cfg_base = {
        "prey_channel_index": int(prey_channel_index),
        "predator_channel_index": 1,  # default complement; per-policy override below
        "dual_traces": True,  # enable both predator and prey memory fields
    }
    if dft_params:
        for k, v in dft_params.items():
            if not str(k).startswith("dft_"):
                custom_cfg_base[f"dft_{k}"] = v
            else:
                custom_cfg_base[k] = v

    rl_module_specs = {}
    for policy_id in obs_keys:
        obs_space = obs_spaces_by_policy[policy_id]
        act_space = act_spaces_by_policy[policy_id]

        # Derive conv filters to fully cover the obs window RF (same as build_module_spec)
        C, H, W = obs_space.shape
        assert H == W and H % 2 == 1, "Expected odd square obs windows (e.g., 7x7, 9x9)."
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

        # Choose channel per policy:
        #  - predators memorize prey trace (channel=2)
        #  - prey memorize predator trace (channel=1)
        role = "prey" if policy_id.endswith("_prey") else ("predator" if policy_id.endswith("_predator") else "unknown")
        if role == "predator":
            chosen_channel = 2
            trace_label = "prey"
        elif role == "prey":
            chosen_channel = 1
            trace_label = "predator"
        else:
            chosen_channel = int(prey_channel_index)
            trace_label = f"ch{chosen_channel}"

        custom_cfg = dict(custom_cfg_base)
        custom_cfg["prey_channel_index"] = chosen_channel
        # Ensure the second trace uses the complementary channel
        if chosen_channel == 2:
            custom_cfg["predator_channel_index"] = 1
        elif chosen_channel == 1:
            custom_cfg["predator_channel_index"] = 2

        # Debug/trace log
        rf = 1 + 2 * L
        conv_str = ",".join(str(c) for c in channels)
        print(
            f"[MODEL] {policy_id} (DFT, trace={trace_label}@ch{chosen_channel}) → obs CxHxW={C}x{H}x{W}, "
            f"L={L} (RF={rf}), conv=[{conv_str}], "
            f"actions={num_actions}, head={head_note}"
        )

        rl_module_specs[policy_id] = RLModuleSpec(
            module_class=DefaultPPOTorchRLModule,
            observation_space=obs_space,
            action_space=act_space,
            inference_only=False,
            model_config={
                # Provide conv/FC to satisfy the new API Catalog for small obs (e.g., 7x7/9x9)
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "fcnet_activation": "relu",
                # Use our custom model that augments obs with a DFT memory channel internally
                "custom_model": "DFTMemoryConvModel",
                "custom_model_config": custom_cfg,
            },
        )

    return MultiRLModuleSpec(rl_module_specs=rl_module_specs)
