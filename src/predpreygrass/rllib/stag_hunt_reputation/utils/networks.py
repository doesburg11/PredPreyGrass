# /mnt/data/networks.py

from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule


def build_module_spec(
    obs_space,
    act_space,
    policy_name: str = None,
    preset: str = "auto",              # NEW: "auto" (default) or "tiny"
    override_model_config: dict = None # NEW: if provided, takes precedence
):
    """
    Build an RLModuleSpec.
    Default behavior ("auto") keeps your current automated setup.
    """

    C, H, W = obs_space.shape
    assert H == W and H % 2 == 1, "Expected odd square obs windows (e.g., 7x7, 9x9)."

    # -------------------------
    # 1) Your existing AUTO logic (UNCHANGED)
    # -------------------------
    L = (H - 1) // 2
    base_channels = [16, 32, 64]
    if L <= len(base_channels):
        channels = base_channels[:L]
    else:
        channels = base_channels + [64] * (L - len(base_channels))

    auto_conv_filters = [[c, [3, 3], 1] for c in channels]

    num_actions = None
    action_info = "None"
    if hasattr(act_space, "n"):
        num_actions = act_space.n
        action_info = str(num_actions)
    elif hasattr(act_space, "nvec"):
        nvec = list(act_space.nvec)
        num_actions = int(sum(nvec))
        action_info = f"nvec={nvec} (sum={num_actions})"

    if num_actions is not None and num_actions > 20:
        auto_fcnet_hiddens = [384, 256]
        head_note = "wide"
    else:
        auto_fcnet_hiddens = [256, 256]
        head_note = "standard"

    # -------------------------
    # 2) Choose config: override > preset > auto
    # -------------------------
    if override_model_config is not None:
        model_config = dict(override_model_config)
        mode_note = "override_model_config"
    elif preset == "tiny":
        # Your experimental "small net"
        model_config = {
            "conv_filters": [
                [8, [3, 3], 1],
                [16, [3, 3], 1],
            ],
            "fcnet_hiddens": [128],
            "fcnet_activation": "relu",
            "vf_share_layers": True,
        }
        mode_note = "preset=tiny"
    else:
        # Default: keep automation
        model_config = {
            "conv_filters": auto_conv_filters,
            "fcnet_hiddens": auto_fcnet_hiddens,
            "fcnet_activation": "relu",
        }
        mode_note = f"preset=auto (head={head_note})"

    # ---- Debug/trace log (once per policy) ----
    if policy_name is not None:
        rf = 1 + 2 * L
        conv_str = ",".join(str(c) for c in channels)
        print(
            f"[MODEL] {policy_name} → obs CxHxW={C}x{H}x{W}, "
            f"L={L} (RF={rf}), auto_conv=[{conv_str}], actions={action_info}, {mode_note}"
        )
        print(f"[MODEL] {policy_name} → model_config={model_config}")

    return RLModuleSpec(
        module_class=DefaultPPOTorchRLModule,
        observation_space=obs_space,
        action_space=act_space,
        inference_only=False,
        model_config=model_config,
    )


def build_multi_module_spec(
    obs_spaces_by_policy: dict,
    act_spaces_by_policy: dict,
    preset: str = "auto",              # NEW
    override_model_config: dict = None # NEW
) -> MultiRLModuleSpec:
    obs_keys = set(obs_spaces_by_policy.keys())
    act_keys = set(act_spaces_by_policy.keys())
    if obs_keys != act_keys:
        missing_in_act = sorted(obs_keys - act_keys)
        missing_in_obs = sorted(act_keys - obs_keys)
        raise ValueError(
            f"Policy key mismatch. Missing in act: {missing_in_act}; Missing in obs: {missing_in_obs}"
        )

    rl_module_specs = {}
    for policy_id in obs_keys:
        rl_module_specs[policy_id] = build_module_spec(
            obs_spaces_by_policy[policy_id],
            act_spaces_by_policy[policy_id],
            policy_name=policy_id,
            preset=preset,
            override_model_config=override_model_config,
        )

    return MultiRLModuleSpec(rl_module_specs=rl_module_specs)
