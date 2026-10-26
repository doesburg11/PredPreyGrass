"""Evaluate fixed social preferences against a frozen existing PPO checkpoint.

Run: python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.evaluate /path/to/checkpoint --episodes 5
This is an inference-only proof of concept, not an evolutionary experiment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ray.rllib.core.columns import Columns

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules, checkpoint_env_config, load_gene_pool, use_gene_pool
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous, SPECIES, SELF_SIZE, SLOT_SIZE
from .social_bias import biased_logits, probabilities


def evaluate(checkpoint, preference, episodes, strength, seed):
    modules = load_modules(checkpoint)
    config = checkpoint_env_config(checkpoint)
    env = PredPreyGrassContinuous(config)
    use_gene_pool(env, load_gene_pool(checkpoint))
    results = []
    for episode in range(episodes):
        rng = np.random.default_rng(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        approach, visible, distances, actions, decisions = 0.0, 0, [], 0, 0
        while obs:
            chosen = {}
            for species in SPECIES:
                ids = [agent for agent in obs if agent.startswith(species)]
                if not ids:
                    continue
                batch = torch.from_numpy(np.stack([np.asarray(obs[a], dtype=np.float32) for a in ids]))
                with torch.inference_mode():
                    logits = modules[species].forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS].detach().cpu().numpy()
                for agent, raw in zip(ids, logits):
                    score = np.asarray(raw, dtype=float)
                    if species == "predator":
                        slots = config.get("predator_observe_count", 1)
                        score = biased_logits(score, obs[agent], preference, strength, slots)
                        for j in range(slots):
                            slot = np.asarray(obs[agent])[SELF_SIZE + j * SLOT_SIZE: SELF_SIZE + (j + 1) * SLOT_SIZE]
                            if len(slot) == SLOT_SIZE and slot[0] > 0.5:
                                distances.append(float(slot[3]))
                                visible += 1
                                break
                    chosen[agent] = int(rng.choice(len(score), p=probabilities(score)))
                    if species == "predator":
                        decisions += 1
                        if chosen[agent] < 32:
                            # Directional alignment of selected moving action with conspecific.
                            from .social_bias import social_scores
                            approach += social_scores(obs[agent], slots)[chosen[agent]]
                        actions += int(chosen[agent] < 32)
            obs, _, terminated, truncated, _ = env.step(chosen)
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        results.append({"episode": episode, "mean_visible_conspecific_distance_normalized": float(np.mean(distances)) if distances else None, "visible_decisions": visible, "mean_alignment_per_predator_decision": approach / max(decisions, 1), "moving_action_fraction": actions / max(decisions, 1)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    result = {str(p): evaluate(args.checkpoint, p, args.episodes, args.strength, args.seed) for p in (-1.0, 0.0, 1.0)}
    output = json.dumps({"checkpoint": str(args.checkpoint), "strength": args.strength, "seed": args.seed, "results": result}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
