"""
Hunt (2006) model-fitting: Stasis / URW / GRW, fit to a single evolving
lineage's per-generation trait-mean time series, selected via AICc.

Reference: Hunt, G. (2006). "Fitting and comparing models of phyletic
evolution: random walks and beyond." Paleobiology, 32(4), 578-601.
Builds on the neutral quantitative-genetics drift null model of Lande, R.
(1976). "Natural selection and random genetic drift in phenotypic
evolution." Evolution, 30(2), 314-334.

This is the formal version of the neutral-drift control this trial family's
own goal statement calls for (README.md: "checked against a neutral-drift
control ... before being trusted as real selection"), applicable directly to
the per-generation `{species}_{trait}_mean` / `_std` / `_count` metrics every
`eco_evolutionary_*` module already logs to `result.json` -- no new
instrumentation, and no need for multiple replicate seeds: it draws its
statistical power from the within-run trajectory (hundreds of generations)
rather than from a small number of seeds.

Models (all fit by exact maximum likelihood on the joint distribution of
sample means, accounting for both within-generation sampling error and the
random-walk covariance between generations):

- Stasis: population mean fluctuates around a constant theta with variance
  omega, i.i.d. per generation. No accumulating change over time.
- URW (Unbiased Random Walk): mean performs a random walk with zero
  directional trend, step variance `vstep`. This is the null model for
  "genetic drift" in the population-genetics sense: unbiased accumulation
  of variance over generations, no selection.
- GRW (General/directional Random Walk): a URW plus a nonzero per-generation
  trend `mstep`. A nonzero, well-supported `mstep` is the signature of
  directional selection.

GRW/URW likelihood is computed via a Kalman filter recursion rather than the
textbook dense joint-covariance formula (Cov[i,j] = vstep*min(t_i,t_j)). They
are mathematically identical, but the dense form is a near-singular matrix
once there are more than a few dozen time points -- LAPACK's solve/slogdet on
it can hit denormal-float slow paths (a single 200x200 slogdet call measured
at ~0.3s instead of microseconds during development of this module). The
Kalman recursion computes the exact same likelihood in O(n) scalar
operations with none of that instability.

Usage: pass four parallel arrays -- generation index, trait mean, trait
variance (std**2), and sample size (population count) per logged
generation/iteration -- to `fit_all_models`. It returns each model's fitted
params, log-likelihood, AICc, and Akaike weight; the model with the lowest
AICc is the best-supported explanation for the observed trajectory. A
well-supported GRW with nonzero `mstep` is evidence of directional
selection; URW winning (or GRW's `mstep` being negligible) is evidence the
trajectory is explained by drift alone.
"""

from __future__ import annotations

import os

# Must precede `import numpy` in this process. Irrelevant to correctness
# (the Kalman recursion below never touches BLAS/LAPACK), kept only because
# this project has separately hit OpenBLAS thread-livelock under sandboxed
# CPU quotas elsewhere; costs nothing to set defensively.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class FitResult:
    model: str
    params: dict
    log_likelihood: float
    n_params: int
    aicc: float


def _aicc(log_likelihood: float, n_params: int, n_obs: int) -> float:
    aic = -2.0 * log_likelihood + 2.0 * n_params
    denom = n_obs - n_params - 1
    if denom <= 0:
        return float(np.inf)
    return aic + (2.0 * n_params * (n_params + 1)) / denom


def _grw_loglik(
    t: np.ndarray, mm: np.ndarray, sampling_var: np.ndarray, anc: float, mstep: float, vstep: float
) -> float:
    """Exact log-likelihood of a (possibly directional) random walk observed with
    per-generation sampling noise, via a scalar Kalman filter (t[0] must be 0)."""
    n = len(t)
    mean_filt = anc
    var_filt = 0.0
    loglik = 0.0
    for i in range(n):
        if i == 0:
            mean_pred, var_pred = anc, 0.0
        else:
            dt = t[i] - t[i - 1]
            mean_pred = mean_filt + mstep * dt
            var_pred = var_filt + vstep * dt
        innovation = mm[i] - mean_pred
        innovation_var = var_pred + sampling_var[i]
        if innovation_var <= 0:
            return float(-np.inf)
        loglik += -0.5 * (np.log(2.0 * np.pi) + np.log(innovation_var) + innovation**2 / innovation_var)
        gain = var_pred / innovation_var
        mean_filt = mean_pred + gain * innovation
        var_filt = (1.0 - gain) * var_pred
    return loglik


def fit_urw(t: np.ndarray, mm: np.ndarray, sampling_var: np.ndarray) -> FitResult:
    """Unbiased random walk: mean(t) = anc, step variance vstep, mstep fixed at 0."""

    def neg_ll(params):
        anc, log_vstep = params
        ll = _grw_loglik(t, mm, sampling_var, anc, 0.0, np.exp(log_vstep))
        return -ll if np.isfinite(ll) else 1e10

    anc0 = float(mm[0])
    vstep0 = max(np.var(np.diff(mm)) / max(np.mean(np.diff(t)), 1e-9), 1e-8)
    res = minimize(neg_ll, x0=[anc0, np.log(vstep0)], method="Nelder-Mead")
    anc, log_vstep = res.x
    vstep = float(np.exp(log_vstep))
    ll = -res.fun
    return FitResult("URW", {"anc": float(anc), "vstep": vstep}, ll, 2, _aicc(ll, 2, len(mm)))


def fit_grw(t: np.ndarray, mm: np.ndarray, sampling_var: np.ndarray) -> FitResult:
    """General (directional) random walk: mean(t) = anc + mstep*t, step variance vstep."""

    def neg_ll(params):
        anc, mstep, log_vstep = params
        ll = _grw_loglik(t, mm, sampling_var, anc, mstep, np.exp(log_vstep))
        return -ll if np.isfinite(ll) else 1e10

    anc0 = float(mm[0])
    mstep0 = (float(mm[-1]) - float(mm[0])) / max(t[-1] - t[0], 1e-9)
    vstep0 = max(np.var(np.diff(mm)) / max(np.mean(np.diff(t)), 1e-9), 1e-8)
    res = minimize(neg_ll, x0=[anc0, mstep0, np.log(vstep0)], method="Nelder-Mead")
    anc, mstep, log_vstep = res.x
    vstep = float(np.exp(log_vstep))
    ll = -res.fun
    return FitResult(
        "GRW", {"anc": float(anc), "mstep": float(mstep), "vstep": vstep}, ll, 3, _aicc(ll, 3, len(mm))
    )


def fit_stasis(mm: np.ndarray, sampling_var: np.ndarray) -> FitResult:
    """Stasis: mean fluctuates around constant theta with variance omega, i.i.d. per generation."""

    def neg_ll(params):
        theta, log_omega = params
        omega = np.exp(log_omega)
        var_i = omega + sampling_var
        resid = mm - theta
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(var_i) + resid**2 / var_i)
        return -ll if np.isfinite(ll) else 1e10

    theta0 = float(np.mean(mm))
    omega0 = max(float(np.var(mm)), 1e-8)
    res = minimize(neg_ll, x0=[theta0, np.log(omega0)], method="Nelder-Mead")
    theta, log_omega = res.x
    omega = float(np.exp(log_omega))
    ll = -res.fun
    return FitResult("Stasis", {"theta": float(theta), "omega": omega}, ll, 2, _aicc(ll, 2, len(mm)))


def fit_all_models(
    generation: np.ndarray, trait_mean: np.ndarray, trait_var: np.ndarray, sample_n: np.ndarray
) -> list[FitResult]:
    """Fit Stasis/URW/GRW to one trait trajectory; return results sorted best (lowest AICc) first."""
    t = np.asarray(generation, dtype=float)
    t = t - t[0]
    mm = np.asarray(trait_mean, dtype=float)
    sampling_var = np.asarray(trait_var, dtype=float) / np.maximum(np.asarray(sample_n, dtype=float), 1.0)

    results = [fit_stasis(mm, sampling_var), fit_urw(t, mm, sampling_var), fit_grw(t, mm, sampling_var)]
    results.sort(key=lambda r: r.aicc)

    best_aicc = results[0].aicc
    weights_raw = [np.exp(-0.5 * (r.aicc - best_aicc)) for r in results]
    total = sum(weights_raw)
    for r, w in zip(results, weights_raw):
        r.params["akaike_weight"] = float(w / total)
    return results


def summarize(results: list[FitResult]) -> str:
    lines = [f"{'model':<8}{'aicc':<12}{'akaike_wt':<12}{'params'}"]
    for r in results:
        wt = r.params.get("akaike_weight", float("nan"))
        other = {k: round(v, 5) for k, v in r.params.items() if k != "akaike_weight"}
        lines.append(f"{r.model:<8}{r.aicc:<12.3f}{wt:<12.4f}{other}")
    return "\n".join(lines)


def load_generation_series(
    result_json, mean_key: str, std_key: str, count_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read (generation, trait_mean, trait_var, sample_n) from a Ray Tune result.json,
    skipping iterations where any of the three metrics is missing or NaN."""
    import json

    gen, mean, var, n = [], [], [], []
    with open(result_json) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            er = d.get("env_runners", {}) or {}
            m, s, c = er.get(mean_key), er.get(std_key), er.get(count_key)
            if m is None or s is None or c is None:
                continue
            if any(isinstance(x, float) and np.isnan(x) for x in (m, s, c)):
                continue
            gen.append(i)
            mean.append(float(m))
            var.append(float(s) ** 2)
            n.append(max(float(c), 1.0))
    return np.array(gen), np.array(mean), np.array(var), np.array(n)


def fit_run(result_json, mean_key: str, std_key: str, count_key: str) -> list[FitResult] | None:
    """Convenience wrapper: load a trait series from result_json and fit all three models.
    Returns None if fewer than 10 generations of data are available (too little for a
    meaningful fit)."""
    gen, mean, var, n = load_generation_series(result_json, mean_key, std_key, count_key)
    if len(gen) < 10:
        return None
    return fit_all_models(gen, mean, var, n)


def report_hunt_fits(
    real_runs: dict[int, object],
    control_runs: dict[int, object],
    trait: str,
    species_list: tuple[str, ...] = ("predator", "prey"),
    prefix: str = "live_genome",
) -> None:
    """Print, per seed and species, the AICc-best of {Stasis, URW, GRW} fit to the full
    generation-by-generation `{species}_{trait}_mean` trajectory in each run's result.json.
    A well-supported GRW with a nonzero `mstep` is the signature of directional selection;
    URW winning (or GRW's `mstep` landing near 0) means the trajectory is explained by drift
    alone. Unlike the Mann-Whitney comparison this project's analyze_replication_seeds.py
    scripts otherwise run, this draws its statistical power from each run's own trajectory
    (hundreds of generations), not from the number of seeds -- see this module's docstring
    and Hunt (2006)."""
    print("=== Hunt (2006) model-fit: drift (URW) vs. directional selection (GRW) vs. stasis ===")
    print(f"Per-seed, per-species AICc-best model fit to the full {trait} generation trajectory.")
    print("A well-supported GRW with nonzero mstep is directional selection; URW (or GRW with")
    print("mstep ~ 0) means the trajectory is explained by drift alone. See model_selection.py.\n")

    species_width = max(10, max(len(s) for s in species_list) + 2)
    header = f"{'group':<8}{'seed':<6}{'species':<{species_width}}{'n_gen':<7}{'best':<8}{'wt':<7}{'GRW mstep'}"
    print(header)
    for group_name, runs in (("real", real_runs), ("control", control_runs)):
        for seed, result_json in sorted(runs.items()):
            for species in species_list:
                gen, mean, var, n = load_generation_series(
                    result_json,
                    f"{prefix}/{species}_{trait}_mean",
                    f"{prefix}/{species}_{trait}_std",
                    f"{prefix}/{species}_count",
                )
                if len(gen) < 10:
                    print(f"{group_name:<8}{seed:<6}{species:<{species_width}} not enough generations logged")
                    continue
                results = fit_all_models(gen, mean, var, n)
                best = results[0]
                grw = next(r for r in results if r.model == "GRW")
                print(
                    f"{group_name:<8}{seed:<6}{species:<{species_width}}{len(gen):<7}{best.model:<8}"
                    f"{best.params['akaike_weight']:<7.3f}{grw.params['mstep']:+.6f}"
                )
    print()
