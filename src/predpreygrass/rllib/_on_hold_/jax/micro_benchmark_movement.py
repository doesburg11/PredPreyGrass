import time, math, argparse
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except Exception as e:
    raise SystemExit("JAX not available: install with pip install --upgrade jax[cpu]")

# Replicate 3x3 action map from env for type_1 agents
_DEF_RANGE = 3
_DELTA = (_DEF_RANGE - 1)//2
ACTION_MAP = {i:(dx,dy) for i,(dx,dy) in enumerate((dx,dy) for dx in range(-_DELTA,_DELTA+1) for dy in range(-_DELTA,_DELTA+1))}
ACTION_DELTAS_JAX = jnp.array([ACTION_MAP[i] for i in range(_DEF_RANGE**2)], dtype=jnp.int32)

# Parameters (mirroring defaults)
ENERGY_DECAY_PRED = 0.15
ENERGY_DECAY_PREY = 0.05
MOVE_ENERGY_FACTOR = 0.01
GRID_SIZE = 25  # larger grid to reduce clipping bias in benchmark


def python_loop_step(positions, energies, is_pred_mask, rng):
    # Decay first (same ordering as env)
    energies -= np.where(is_pred_mask, ENERGY_DECAY_PRED, ENERGY_DECAY_PREY)
    # Random actions
    acts = rng.integers(0, _DEF_RANGE**2, size=positions.shape[0])
    # Apply movement + clip + collision ignore (no collision check to match JAX approximation)
    deltas = np.array([ACTION_MAP[a] for a in acts], dtype=np.int32)
    new_positions = positions + deltas
    np.clip(new_positions, 0, GRID_SIZE-1, out=new_positions)
    disp = new_positions - positions
    dist = np.sqrt((disp*disp).sum(axis=1))
    move_cost = dist * MOVE_ENERGY_FACTOR * energies
    energies -= move_cost
    return new_positions, energies

@jax.jit
def jax_step(positions, energies, is_pred_mask, key):
    # key split
    key, sub = jax.random.split(key)
    acts = jax.random.randint(sub, (positions.shape[0],), 0, _DEF_RANGE**2)
    deltas = ACTION_DELTAS_JAX[acts]
    decay = is_pred_mask * ENERGY_DECAY_PRED + (1 - is_pred_mask) * ENERGY_DECAY_PREY
    energies_after_decay = energies - decay
    new_positions = positions + deltas
    new_positions = jnp.clip(new_positions, 0, GRID_SIZE-1)
    disp = new_positions - positions
    dist = jnp.sqrt(jnp.sum(disp*disp, axis=1))
    move_cost = dist * MOVE_ENERGY_FACTOR * energies_after_decay
    new_energies = energies_after_decay - move_cost
    return new_positions, new_energies, key


def bench(mode, steps, n_agents, predator_frac=0.43, warmup=50):
    n_pred = int(n_agents * predator_frac)
    n_prey = n_agents - n_pred
    rng = np.random.default_rng(123)
    positions = rng.integers(0, GRID_SIZE, size=(n_agents,2), dtype=np.int32)
    energies = rng.random(n_agents).astype(np.float32) * 5 + 1.0
    is_pred_mask_np = np.zeros(n_agents, dtype=np.int32)
    is_pred_mask_np[:n_pred] = 1
    rng.shuffle(is_pred_mask_np)

    if mode == 'python':
        for _ in range(warmup):
            positions, energies = python_loop_step(positions, energies, is_pred_mask_np, rng)
        start = time.perf_counter()
        for _ in range(steps):
            positions, energies = python_loop_step(positions, energies, is_pred_mask_np, rng)
        elapsed = time.perf_counter() - start
        return steps/elapsed, elapsed
    else:
        # JAX mode
        positions_j = jnp.array(positions)
        energies_j = jnp.array(energies)
        is_pred_j = jnp.array(is_pred_mask_np)
        key = jax.random.PRNGKey(123)
        # Warmup (compile + a few runs)
        for _ in range(warmup):
            positions_j, energies_j, key = jax_step(positions_j, energies_j, is_pred_j, key)
        # Force compilation completion
        jax.block_until_ready(positions_j)
        start = time.perf_counter()
        for _ in range(steps):
            positions_j, energies_j, key = jax_step(positions_j, energies_j, is_pred_j, key)
        jax.block_until_ready(positions_j)
        elapsed = time.perf_counter() - start
        return steps/elapsed, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--agents', type=int, nargs='+', default=[14, 50, 100, 200])
    ap.add_argument('--warmup', type=int, default=100)
    args = ap.parse_args()

    print("Micro-benchmark: movement + decay (Python vs JAX)\n")
    print(f"Steps per trial: {args.steps}, Warmup: {args.warmup}\n")
    header = f"{'Agents':>8}  {'Python sps':>12}  {'JAX sps':>12}  {'Speedup x':>10}"
    print(header)
    print('-'*len(header))
    for n in args.agents:
        py_sps, py_elapsed = bench('python', args.steps, n, warmup=args.warmup)
        jx_sps, jx_elapsed = bench('jax', args.steps, n, warmup=args.warmup)
        speedup = jx_sps / py_sps
        print(f"{n:8d}  {py_sps:12.1f}  {jx_sps:12.1f}  {speedup:10.2f}")

    print("\nNOTE: This isolates only the vectorized piece (decay + movement for type_1 logic analogue).\nActual end-to-end env speedup will be smaller due to reproduction, engagement, observation building still in Python.")

if __name__ == '__main__':
    main()
