"""The predators' PPO network with the innate bias (DESIGN.md, three arms):
RLlib's default PPO module, whose action logits get genes.bias added inside
every forward pass (exploration, inference and training). Actions, their
stored log-probabilities, PPO's ratios, entropy and KL therefore all come
from the same net + bias distribution, so PPO stays on-policy. The value
function sees the same observation (genes included) and is not changed.

model_config["gene_bias"] = {"beta": float, "layout": genes.layout(env)}.
With beta 0 (arm L) the module is RLlib's default one.
"""

import torch

from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.core.columns import Columns

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.genes import DIRECTION_TABLE, GENES, SLOT_SIZE, SPEED_TABLE


def _mean_unit(slots, mask):
    """Mean unit offset over the masked slots; zero if none (torch)."""
    offsets = slots[..., 1:3]
    norm = offsets.norm(dim=-1, keepdim=True)
    unit = torch.where(norm > 0, offsets / norm.clamp_min(1e-12), torch.zeros_like(offsets))
    weight = mask.unsqueeze(-1).to(offsets.dtype)
    count = weight.sum(dim=-2)
    return (unit * weight).sum(dim=-2) / count.clamp_min(1.0)


def torch_bias(obs, lay, beta, directions, speeds):
    """(n, MOVES) innate bias from (n, obs_size) observations (genes.bias in
    torch)."""
    n = obs.shape[0]
    hunters = obs[:, lay["hunter_start"] : lay["hunter_start"] + SLOT_SIZE * lay["hunter_count"]].reshape(n, lay["hunter_count"], SLOT_SIZE)
    deer = obs[:, lay["deer_start"] : lay["deer_start"] + SLOT_SIZE * lay["deer_count"]].reshape(n, lay["deer_count"], SLOT_SIZE)
    energy = obs[:, lay["deer_energy_start"] : lay["deer_energy_start"] + lay["deer_count"]]
    genes = obs[:, lay["gene_start"] : lay["gene_start"] + len(GENES)]
    seen_deer = deer[..., 0] > 0.5
    adult = seen_deer & (energy >= lay["adult_threshold"])
    young = seen_deer & ~adult
    d = genes[:, 0:1] * _mean_unit(hunters, hunters[..., 0] > 0.5) + genes[:, 1:2] * _mean_unit(deer, young) + genes[:, 2:3] * _mean_unit(deer, adult)
    return beta * speeds.unsqueeze(0) * (d @ directions.T)


class GeneBiasPPOTorchRLModule(DefaultPPOTorchRLModule):
    def setup(self):
        super().setup()
        settings = self.model_config["gene_bias"]
        self.gene_beta = float(settings["beta"])
        self.gene_layout = dict(settings["layout"])
        if self.observation_space.shape != (self.gene_layout["obs_size"],):
            raise ValueError(f"observation shape {self.observation_space.shape} does not fit the gene layout")
        # Not saved with the weights (persistent=False): the state matches
        # RLlib's default module.
        self.register_buffer("gene_directions", torch.tensor(DIRECTION_TABLE, dtype=torch.float32), persistent=False)
        self.register_buffer("gene_speeds", torch.tensor(SPEED_TABLE, dtype=torch.float32), persistent=False)

    def _add_bias(self, batch, output):
        if self.gene_beta:
            obs = batch[Columns.OBS]
            logits = output[Columns.ACTION_DIST_INPUTS]
            bias = torch_bias(obs.to(logits.dtype), self.gene_layout, self.gene_beta, self.gene_directions.to(logits), self.gene_speeds.to(logits))
            output[Columns.ACTION_DIST_INPUTS] = logits + bias
        return output

    def _forward(self, batch, **kwargs):
        return self._add_bias(batch, super()._forward(batch, **kwargs))

    def _forward_train(self, batch, **kwargs):
        return self._add_bias(batch, super()._forward_train(batch, **kwargs))
