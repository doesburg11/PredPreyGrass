import gymnasium
import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec


def _apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits for invalid actions to -inf by adding log(mask)."""
    return logits + torch.clamp(torch.log(mask.float() + 1e-9), min=-1e9)


class ActionMaskedPPOModule(DefaultPPOTorchRLModule):
    """PPO RLModule with action masking.

    Expects observations as Dict{"observations": Box(C,H,W), "action_mask": Box(n_actions,)}.
    The CNN encoder processes only "observations"; the mask is applied to policy logits
    before sampling, so invalid actions (frozen steps) get probability zero.
    """

    @staticmethod
    def _unwrap_observations(batch):
        if isinstance(batch.get("obs"), dict) and "observations" in batch["obs"]:
            inner = dict(batch)
            inner["obs"] = batch["obs"]["observations"]
            return inner
        return batch

    def _forward_train(self, batch, **kwargs):
        mask = batch["obs"]["action_mask"]
        out = super()._forward_train(self._unwrap_observations(batch), **kwargs)
        out["action_dist_inputs"] = _apply_action_mask(out["action_dist_inputs"], mask)
        return out

    def _forward_inference(self, batch, **kwargs):
        mask = batch["obs"]["action_mask"]
        out = super()._forward_inference(self._unwrap_observations(batch), **kwargs)
        out["action_dist_inputs"] = _apply_action_mask(out["action_dist_inputs"], mask)
        return out

    def _forward_exploration(self, batch, **kwargs):
        mask = batch["obs"]["action_mask"]
        out = super()._forward_exploration(self._unwrap_observations(batch), **kwargs)
        out["action_dist_inputs"] = _apply_action_mask(out["action_dist_inputs"], mask)
        return out

    def compute_values(self, batch, embeddings=None):
        return super().compute_values(self._unwrap_observations(batch), embeddings=embeddings)


def build_module_spec(obs_space, act_space, policy_name: str = None):
    """
    Build an RLModuleSpec for ActionMaskedPPOModule.

    obs_space may be a Dict{"observations": Box(C,H,W), "action_mask": Box(n,)}.
    The CNN is built from the spatial "observations" part only.
    """
    inner_obs = obs_space["observations"] if isinstance(obs_space, gymnasium.spaces.Dict) else obs_space
    C, H, W = inner_obs.shape

    assert H == W and H % 2 == 1, "Expected odd square obs windows (e.g., 7x7, 9x9)."

    L = (H - 1) // 2
    base_channels = [16, 32, 64]
    channels = base_channels[:L] if L <= len(base_channels) else base_channels + [64] * (L - len(base_channels))
    conv_filters = [[c, [3, 3], 1] for c in channels]

    num_actions = act_space.n if hasattr(act_space, "n") else None
    fcnet_hiddens = [384, 256] if (num_actions is not None and num_actions > 20) else [256, 256]

    if policy_name is not None:
        rf = 1 + 2 * L
        conv_str = ",".join(str(c) for c in channels)
        print(
            f"[MODEL] {policy_name} → obs CxHxW={C}x{H}x{W}, "
            f"L={L} (RF={rf}), conv=[{conv_str}], actions={num_actions}"
        )

    return RLModuleSpec(
        module_class=ActionMaskedPPOModule,
        observation_space=inner_obs,
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
    obs_keys = set(obs_spaces_by_policy.keys())
    act_keys = set(act_spaces_by_policy.keys())
    if obs_keys != act_keys:
        raise ValueError(f"Policy key mismatch: obs={obs_keys}, act={act_keys}")
    return MultiRLModuleSpec(rl_module_specs={
        pid: build_module_spec(obs_spaces_by_policy[pid], act_spaces_by_policy[pid], policy_name=pid)
        for pid in obs_keys
    })
