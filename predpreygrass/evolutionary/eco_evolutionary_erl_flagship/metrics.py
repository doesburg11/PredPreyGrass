"""CSV logging for Trial 13, re-exported unchanged from eco_evolutionary_erl_baldwin/
metrics.py: `CsvLogger`, `lineage_fieldnames`, `lineage_record`, and
`truncate_csv_after_step` only depend on an agent-like object exposing
`.agent_id`, `.generation`, `.born_step`, `.offspring_count`, and `.genome`
(eval_weights/eval_bias) -- exactly what driver.py's `PreyGenomeState` provides --
plus `obs_dim`, so they apply verbatim with obs_dim=8.

FunctionalConstraintTracker is NOT re-exported: it's Trial 12's genetic-assimilation
detection method (tracking eval- vs. action-network site drift across generations),
orthogonal to this module's proximate-vs-ultimate-reward question and not currently
used by run_trial13_simulation.py. Import directly from erl_baldwin if a future
stage needs it.
"""

from predpreygrass.evolutionary.eco_evolutionary_erl_baldwin.metrics import (  # noqa: F401
    CsvLogger,
    lineage_fieldnames,
    lineage_record,
    truncate_csv_after_step,
)
