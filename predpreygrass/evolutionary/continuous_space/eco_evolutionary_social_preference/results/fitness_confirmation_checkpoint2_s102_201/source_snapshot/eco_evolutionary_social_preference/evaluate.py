"""Evaluate fixed social preferences against a frozen existing PPO checkpoint.

Run: python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.evaluate /path/to/checkpoint --episodes 5
This is an inference-only proof of concept, not an evolutionary experiment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ray.rllib.core.columns import Columns

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.checkpoints import load_modules, checkpoint_env_config, load_gene_pool, use_gene_pool
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.environment import PredPreyGrassContinuous, SPECIES, SELF_SIZE, SLOT_SIZE
from .social_bias import biased_logits, social_scores, probabilities
from .validation import validate_checkpoint
from .movement_metrics import MovementMetrics, capture
from .summarize import summarize
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.eval import select_actions


def evaluate(checkpoint, preference, episodes, strength, seed, modules=None, on_episode=None, observer=None, individual_preferences=None):
    modules = load_modules(checkpoint) if modules is None else modules
    config = checkpoint_env_config(checkpoint)
    env = PredPreyGrassContinuous(config)
    use_gene_pool(env, load_gene_pool(checkpoint))
    results = []
    neutral_reference_check = preference == 0 if individual_preferences is None else individual_preferences.is_uniform_neutral
    if observer is not None or individual_preferences is not None:
        original_physics_step = env._physics_step
        def observed_physics_step(*args, **kwargs):
            transition = original_physics_step(*args, **kwargs)
            if individual_preferences is not None:
                individual_preferences.register(env)
            if observer is not None:
                observer.sample(env)
            return transition
        env._physics_step = observed_physics_step
    for episode in range(episodes):
        generator = torch.Generator().manual_seed(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        if individual_preferences is not None:
            individual_preferences.start_episode(env, seed + episode)
        if observer is not None:
            observer.reset(env, seed + episode)
        approach, visible, distances, actions, decisions = 0.0, 0, [], 0, 0
        counterfactual = {str(p): [] for p in (-1.0, 0.0, 1.0)}
        visible_moves, expected_alignment, expected_moving, steps = 0, [], [], 0
        movement = MovementMetrics((env.width, env.height), env.distance_scale)
        while obs:
            before = capture(env)
            start_tick = env.time_step
            chosen = {}
            if neutral_reference_check:
                reference_generator = torch.Generator()
                reference_generator.set_state(generator.get_state())
                reference_actions = select_actions(modules, obs, reference_generator, True)
            for species in SPECIES:
                ids = sorted(agent for agent in obs if agent.startswith(species))
                if not ids:
                    continue
                batch = torch.from_numpy(np.stack([np.asarray(obs[a], dtype=np.float32) for a in ids]))
                with torch.inference_mode():
                    logits = modules[species].forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS].detach().cpu().numpy()
                adjusted = logits.copy()
                for index, (agent, raw) in enumerate(zip(ids, logits)):
                    score = np.asarray(raw, dtype=float)
                    if species == "predator":
                        slots = config.get("predator_observe_count", 1)
                        active_preference = preference if individual_preferences is None else individual_preferences.get(agent)
                        score = biased_logits(score, obs[agent], active_preference, strength, slots)
                        adjusted[index] = score
                        for j in range(slots):
                            slot = np.asarray(obs[agent])[SELF_SIZE + j * SLOT_SIZE: SELF_SIZE + (j + 1) * SLOT_SIZE]
                            if len(slot) == SLOT_SIZE and slot[0] > 0.5:
                                distances.append(float(slot[3]))
                                visible += 1
                                break
                # Match the original evaluator's float32 softmax, batched draws,
                # sorted agent order and Torch generator exactly at preference 0.
                probs = torch.softmax(torch.from_numpy(adjusted), dim=-1)
                sampled = torch.multinomial(probs, 1, generator=generator).squeeze(1)
                for index, (agent, action) in enumerate(zip(ids, sampled.tolist())):
                    chosen[agent] = action
                    if species == "predator":
                        alignment = social_scores(obs[agent], slots)
                        has_target = bool(np.any(alignment))
                        moving = action < 32
                        decisions += 1
                        actions += int(moving)
                        if has_target:
                            approach += alignment[action]
                            visible_moves += int(moving)
                            for alternative in (-1.0, 0.0, 1.0):
                                cf = probabilities(biased_logits(logits[index], obs[agent], alternative, strength, slots))
                                counterfactual[str(alternative)].append(float(cf @ alignment / cf[:32].sum()))
                            probability = probs[index].numpy()
                            moving_mass = probability[:32].sum()
                            expected_moving.append(float(moving_mass))
                            expected_alignment.append(float(probability @ alignment / moving_mass) if moving_mass > 0 else 0.0)
            if neutral_reference_check:
                assert chosen == reference_actions, "neutral actions differ from original evaluator"
                assert torch.equal(generator.get_state(), reference_generator.get_state())
            steps += 1
            obs, _, terminated, truncated, _ = env.step(chosen)
            after_positions, _ = capture(env)
            movement.record(before, after_positions, env.time_step - start_tick)
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        results.append({
            "episode": episode, "seed": seed + episode, "steps": steps,
            "predator_decisions": decisions,
            "movement": movement.result(),
            "neutral_original_actions_verified": neutral_reference_check,
            "mean_visible_conspecific_distance_normalized": float(np.mean(distances)) if distances else None,
            "visible_decisions": visible,
            "visible_decision_fraction": visible / max(decisions, 1),
            "mean_alignment_per_predator_decision": approach / max(decisions, 1),
            "moving_action_fraction": actions / max(decisions, 1),
            "visible_moving_decisions": visible_moves,
            "same_observation_expected_alignment_given_moving": {
                key: float(np.mean(values)) if values else None for key, values in counterfactual.items()
            },
            "mean_alignment_given_visible_and_moving": approach / visible_moves if visible_moves else None,
            "moving_fraction_given_visible": visible_moves / visible if visible else None,
            "mean_expected_alignment_given_visible_and_moving": float(np.mean(expected_alignment)) if expected_alignment else None,
            "mean_expected_moving_probability_given_visible": float(np.mean(expected_moving)) if expected_moving else None,
        })
        if observer is not None:
            results[-1]["persistence"] = observer.finish(env)
        if individual_preferences is not None:
            results[-1]["individual_preferences"] = individual_preferences.result(env)
        preference_label = f"{preference:+g}" if individual_preferences is None else individual_preferences.mode
        print(f"preference={preference_label} seed={seed + episode} steps={steps} complete", file=sys.stderr, flush=True)
        if on_episode is not None:
            on_episode(preference, results[-1])
    env.close()
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
    if not np.isfinite(args.strength) or args.strength < 0:
        parser.error("--strength must be finite and nonnegative")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new experiment path")
    torch.set_num_threads(1)
    modules = load_modules(args.checkpoint)
    validation = validate_checkpoint(args.checkpoint, modules, args.seed)
    progress = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        progress = args.output.with_suffix('.episodes.jsonl').open('x')
    def completed(preference, episode):
        if progress:
            progress.write(json.dumps({'preference': preference, **episode}) + '\n')
            progress.flush()
    try:
        result = {str(p): evaluate(args.checkpoint, p, args.episodes, args.strength, args.seed, modules, completed) for p in (-1.0, 0.0, 1.0)}
    finally:
        if progress:
            progress.close()
    validation_after = validate_checkpoint(args.checkpoint, modules, args.seed)
    if validation_after != validation:
        raise RuntimeError("checkpoint validation changed during evaluation")
    output = json.dumps({"checkpoint": str(args.checkpoint), "strength": args.strength, "seed": args.seed, "validation": validation, "validation_after": validation_after, "summary": summarize(result), "results": result}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
