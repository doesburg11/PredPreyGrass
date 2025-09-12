from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog


def _gaussian_kernel2d(ks: int = 3, sigma: float = 1.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a 2D Gaussian kernel with size ks (odd) and std sigma."""
    ks = int(max(3, ks))
    if ks % 2 == 0:
        ks += 1
    ax = torch.arange(ks, dtype=torch.float32, device=device) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


class DFTMemoryConvModel(TorchModelV2, nn.Module):
    """
    TorchModelV2 that maintains a per-agent DFT-like memory field u over the
    agent's local observation window. The memory is driven by the observed prey
    channel and appended as an extra channel before the conv stack.

    State: a single tensor [B, H, W] representing u for each batch element.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Observation shape (C, H, W)
        C, H, W = obs_space.shape
        assert H == W and H % 2 == 1, "Expected odd square obs windows."
        self.C = C
        self.H = H
        self.W = W

        # DFT parameters (defaults aligned with dft_2.py values)
        cfg = model_config.get("custom_model_config", {}) if model_config else {}
        # Channel indices (0:mask,1:pred,2:prey,3:grass)
        self.prey_channel_index = int(cfg.get("prey_channel_index", 2))
        self.predator_channel_index = int(cfg.get("predator_channel_index", 1))
        self.dt = float(cfg.get("dft_dt", 1.0))
        self.tau = float(cfg.get("dft_tau", 10.0))
        self.gamma = float(cfg.get("dft_gamma", 0.55))
        self.exc_sigma = float(cfg.get("dft_exc_sigma", 1.2))
        self.exc_gain = float(cfg.get("dft_exc_gain", 0.60))
        self.input_gain = float(cfg.get("dft_input_gain", 1.60))
        self.zmin = float(cfg.get("dft_zmin", 0.0))
        self.zmax = float(cfg.get("dft_zmax", 1.0))

        # Dual trace support
        self.dual_traces = bool(cfg.get("dual_traces", False))

        # Base DFT parameters (trace #1)
        ksize = int(cfg.get("dft_kernel_size", 5))
        kernel = _gaussian_kernel2d(ksize, sigma=max(0.5, self.exc_sigma))
        self.register_buffer("gauss_kernel", kernel.unsqueeze(0).unsqueeze(0))  # [1,1,K,K]

        # Optional second trace parameterization (trace #2)
        # Either explicit values (dft2_*) or scales relative to base (dft2_*_scale)
        self.input_gain2 = None
        self.exc_gain2 = None
        self.tau2 = None
        if self.dual_traces:
            if "dft2_input_gain" in cfg:
                self.input_gain2 = float(cfg.get("dft2_input_gain"))
            else:
                self.input_gain2 = float(cfg.get("dft2_input_gain_scale", 0.8)) * self.input_gain
            if "dft2_exc_gain" in cfg:
                self.exc_gain2 = float(cfg.get("dft2_exc_gain"))
            else:
                self.exc_gain2 = float(cfg.get("dft2_exc_gain_scale", 0.5)) * self.exc_gain
            if "dft2_tau" in cfg:
                self.tau2 = float(cfg.get("dft2_tau"))
            else:
                self.tau2 = float(cfg.get("dft2_tau_scale", 0.7)) * self.tau

        # Conv stack inspired by utils.networks.build_module_spec receptive field math
        L = (H - 1) // 2
        base_channels = [16, 32, 64]
        if L <= len(base_channels):
            channels = base_channels[:L]
        else:
            channels = base_channels + [64] * (L - len(base_channels))

        conv_layers = []
        in_ch = self.C + (2 if self.dual_traces else 1)  # + memory channels
        for out_ch in channels:
            conv_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers) if conv_layers else nn.Identity()

        # Compute flat size after conv (still HxW with stride=1 and padding=1)
        conv_out_ch = channels[-1] if channels else (self.C + (2 if self.dual_traces else 1))
        flat_size = conv_out_ch * H * W

        # FC head sizes depending on action count
        num_actions = getattr(action_space, "n", None)
        if num_actions is not None and num_actions > 20:
            fc_sizes = [384, 256]
        else:
            fc_sizes = [256, 256]

        fcs = []
        last = flat_size
        for h in fc_sizes:
            fcs += [nn.Linear(last, h), nn.ReLU(inplace=True)]
            last = h
        self.fc = nn.Sequential(*fcs)

        # Policy logits and value head
        out_size = num_outputs  # RLlib expects num_outputs = action space size
        self.logits_layer = nn.Linear(last, out_size)
        self.value_layer = nn.Linear(last, 1)

        # Placeholder for last value
        self._value_out = None

    # ----- DFT memory update -----
    def _dft_step(self, u: torch.Tensor, stim01: torch.Tensor, *, input_gain: float, exc_gain: float, tau: float) -> torch.Tensor:
        # u: [B,1,H,W]; stim01: [B,1,H,W]
        if self.gauss_kernel is not None:
            exc = exc_gain * F.conv2d(u, self.gauss_kernel, padding=self.gauss_kernel.shape[-1] // 2)
        else:
            exc = exc_gain * u
        global_inh = self.gamma * u.mean(dim=[2, 3], keepdim=True)
        du = (-u + input_gain * stim01 + exc - global_inh) / tau
        u = torch.clamp(u + self.dt * du, self.zmin, self.zmax)
        return u

    # ----- TorchModelV2 API -----
    def get_initial_state(self) -> List[torch.Tensor]:
        # State shape(s): [1, H, W] per field per batch element (RLlib will tile as needed)
        u0 = torch.zeros((1, self.H, self.W), dtype=torch.float32)
        if self.dual_traces:
            v0 = torch.zeros((1, self.H, self.W), dtype=torch.float32)
            return [u0, v0]
        return [u0]

    def forward(self, input_dict, state, seq_lens):
        # obs: [B, C, H, W]
        obs = input_dict["obs"].float()
        B = obs.shape[0]

        # Prepare/resize state
        def _prep_state(t: Optional[torch.Tensor]) -> torch.Tensor:
            if t is None:
                return torch.zeros((B, 1, self.H, self.W), dtype=obs.dtype, device=obs.device)
            s = t
            if s.dim() == 3:
                s = s.unsqueeze(1)
            if s.shape[0] != B:
                if s.shape[0] == 1:
                    s = s.expand(B, -1, -1, -1).contiguous()
                else:
                    s = s[:B]
            return s

        u = _prep_state(state[0] if state else None)
        v = None
        if self.dual_traces:
            v = _prep_state(state[1] if (state and len(state) > 1) else None)

        # Stimulus from configured channels
        prey_idx = max(0, min(self.C - 1, self.prey_channel_index))
        pred_idx = max(0, min(self.C - 1, self.predator_channel_index))
        stim_prey = obs[:, prey_idx:prey_idx + 1, :, :]
        stim_pred = obs[:, pred_idx:pred_idx + 1, :, :]

        # DFT update(s)
        u_next = self._dft_step(u, stim_prey, input_gain=self.input_gain, exc_gain=self.exc_gain, tau=self.tau)
        mem_channels = [u_next]
        state_out: List[torch.Tensor] = [u_next.squeeze(1).detach()]
        if self.dual_traces:
            assert v is not None and self.input_gain2 is not None and self.exc_gain2 is not None and self.tau2 is not None
            v_next = self._dft_step(v, stim_pred, input_gain=self.input_gain2, exc_gain=self.exc_gain2, tau=self.tau2)
            mem_channels.append(v_next)
            state_out.append(v_next.squeeze(1).detach())

        # Augment obs with memory channel(s)
        obs_aug = torch.cat([obs] + mem_channels, dim=1)

        # Feature extractor
        x = self.conv(obs_aug)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # Heads
        logits = self.logits_layer(x)
        self._value_out = self.value_layer(x).squeeze(-1)
        return logits, state_out

    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out


# Register so we can reference by string in model_config
ModelCatalog.register_custom_model("DFTMemoryConvModel", DFTMemoryConvModel)
