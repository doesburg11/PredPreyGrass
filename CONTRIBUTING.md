# Contributing to PredPreyGrass

Thanks for considering a contribution. This is a research codebase first, so the bar for
a change is: does it run, is it tested, and does it fit the evolutionary /
non-evolutionary split described in [EXPERIMENTS.md](EXPERIMENTS.md)?

## Reporting bugs, proposing changes, or asking questions

Use the issue templates — [Bug Report](.github/ISSUE_TEMPLATE/bug.yml),
[Proposal](.github/ISSUE_TEMPLATE/proposal.yml), or
[Question](.github/ISSUE_TEMPLATE/question.yml) — rather than a blank issue; they ask for
the context needed to act on it.

## Setting up your environment

Follow the [Quick start](README.md#quick-start-run-a-demo-in-under-five-minutes) in the
README, then install the dev extras and pre-commit hooks:

```bash
pip install -e .[dev]
pip install pre-commit
pre-commit install
```

`pre-commit` runs `black` and `ruff` on every commit; run it over the whole repo any time
with:

```bash
pre-commit run --all-files
```

## Before opening a pull request

- Run `pytest -v` against the module(s) you touched — most environments carry their own
  `tests/` directory alongside the implementation.
- Run `pre-commit run --all-files` and let it apply fixes.
- Fill out the [pull request template](.github/PULL_REQUEST_TEMPLATE.md) checklist.

## Adding a new environment

New environments live under `predpreygrass/evolutionary/` (a heritable trait under
selection) or `predpreygrass/non_evolutionary/` (fixed traits, incentive-design
variation only) — see [EXPERIMENTS.md](EXPERIMENTS.md) for the distinction and existing
examples to follow. Each environment module is expected to ship its own `README.md`
(methodology/results) and `tests/` directory, matching the pattern of its siblings.

## Repeatable editor checks

Install the Python development dependencies with `.conda/bin/python -m pip install -e '.[dev]'`,
then install the pinned Node tools with `npm ci`. Run `npm run check` from the repository root.

The current clean scope covers the `stag_hunt_forward_view` and `stag_hunt_defection` Python modules, the
archived-source evaluators in `stag_hunt_reputation`, and the forward-view and
defection READMEs plus the stag-hunt comparison. Local Markdown and HTML links are checked across tracked
documentation, excluding the active continuous environment. Existing lint findings
in other experiments are outside this scope. Pyright uses the same `.conda`
environment configured for VS Code; no diagnostic categories are disabled.
