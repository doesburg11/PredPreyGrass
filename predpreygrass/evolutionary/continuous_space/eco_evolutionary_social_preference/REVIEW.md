# Independent Codex CLI review — 2026-10-26

## CLI repair and successful verification

The configured default `gpt-6.1-sol` was rejected by the ChatGPT-authenticated
CLI. A read-only probe using `gpt-5.6-terra` succeeded. The user configuration
was backed up and its default model changed to `gpt-5.6-terra`. The full
read-only review then completed with exit code 0 using the repaired default.
No login credentials, PPO policies, checkpoints, or experiment results changed.

## Findings and primary-agent assessment

- **Confirmed precision mismatch:** `evaluate.py:81` computes same-observation
  counterfactual probabilities through NumPy float64, whereas actions use
  Torch float32 softmax at `evaluate.py:67`. The reviewer calls this medium
  severity. I agree it is not numerically identical, but assess it as a
  precision/documentation issue for these runs, rather than evidence that
  their physical movement results are wrong. Use the same float32 path in
  a future patch or label the counterfactual as an analytic approximation.
- **Confirmed optimized-interpreter gap:** neutral checks at
  `evaluate.py:88` and architecture/configuration checks in
  `validation.py:35` onward use Python assertions, which disappear under
  `python -O`. Replace them with explicit exceptions in a future patch.
  The recorded evaluations used the standard interpreter, so those checks
  did execute. Tensor equality and independent forward-output comparisons
  also use NumPy/Torch testing functions, rather than only Python assertions.
- **Disputed exact-tie claim:** the reviewer says the environment sorts full
  `(distance, dx, dy, animal)` tuples. Actual `environment.py:871` sorts
  with `key=lambda entry: entry[0]`, preserving list order on equal distances.
  Metric capture also preserves list order. Therefore the claimed different
  tie-break is not supported by the code. A symmetric-position regression
  test would still be useful; distance calculations use different numeric
  routines and near-tie rounding remains worth checking.
- **Reproducibility follow-up:** fingerprint gene pools and full settings,
  not just weights and architecture, if checkpoint files might change
  concurrently. The active training jobs used separate seed-3 runs; this
  review found no evidence that the completed seed-1 evaluation was corrupted.
- **Unit labels:** the report converts normalized nearest-neighbor distance
  to world units by multiplying by 400. This conversion is correct; make
  it explicit in metadata or labels to prevent downstream confusion.
- **Coverage follow-ups:** checkpoint-corruption rejection fixtures,
  an automated checkpoint-to-JSONL integration test, float32 counterfactual
  equivalence, and boundary/multiple-slot/short-interval cases.

The reviewer agreed that movement projections, pair-closing definitions,
wrapped distance, death exclusions, and episode-level pairing are correct.
It also agreed that checkpoint tensor comparisons and independent actor
calculations provide strong fidelity checks. Previously documented survivor,
population, time-weighting and single-checkpoint limitations remain.

The reviewer emitted some incorrect file line numbers after concatenating
numbered files. References above were checked against individual source files.
This record summarizes its findings and my assessment; it is not a verbatim
review transcript. The raw CLI transcript is `/tmp/social_movement_review_fixed.log`.

No experiment-code changes were made as part of the CLI repair. The confirmed
follow-ups above remain open; fixing model access does not automatically fix
the reviewed code.


## Persistence review, finalized 2026-10-27

The independent review of `persistence.py`, its tests and the observer hook
found no correctness defects. I agree with its checks of interval exposure,
continuous pair identity, birth/initial/terminal censoring, directed nearest
neighbors and unchanged inference/RNG/physics. Local tests pass (27 tests),
and an observer-enabled two-condition seed-42 smoke run reproduces prior
movement and action metrics.

The first runner/replay review was interrupted by CLI usage limits. A fresh
review completed before this commit. It confirmed that the paired statistics
are correctly keyed by seed and reported these open follow-ups:

1. **Trace-to-run binding:** the runner records input/source hashes but no
   per-trace hashes; the replay trusts filenames and matching initial frames.
   Record trace hashes with condition/seed/checkpoint provenance at evaluation
   completion, and verify them before replay. The reviewer rates this high.
   I agree it is a reproducibility hardening gap for mixed or substituted
   artifacts, rather than evidence that the known traces generated in this
   session are wrong. This commit freezes raw results, traces and rendered
   videos together, but does not retroactively add a recording-time binding.
2. **Treatment validation:** the summarizer checks matching provenance and
   condition labels but does not explicitly require preference 0 versus 1
   and strength 2. Add those checks before accepting arbitrary inputs.
   The reviewer rates this medium. I verified those fields directly in the
   actual two raw files; the intended contrast is correct for these results.
3. **Ambiguous input selection:** `next(glob(...))` silently selects the first
   file when a directory contains multiple runs. Require one match or explicit
   input paths. The reviewer rates this low. I verified exactly one input per
   condition in this experiment directory.

A minor visualization limitation is that trails crossing a torus seam are
omitted rather than split and wrapped. The plotted positions and green
proximity edges are correct. No inference or numerical result depends on
trail rendering.

These follow-ups remain open; the completed report does not claim that
arbitrary external or mixed artifacts are fully validated. The raw transcripts
are `/tmp/social_persistence_review.log` and
`/tmp/persistence_commit_review.log`. Some reviewer test commands used the
system environment without pytest; all 27 tests pass in the project's `.conda`.
