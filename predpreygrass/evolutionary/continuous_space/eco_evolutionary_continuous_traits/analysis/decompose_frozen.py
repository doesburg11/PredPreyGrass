"""Analysis of decompose_frozen.py (RESULTS.md section 7): the change in
mean predator speed in the re-run both-frozen arm, split per episode into

- selection among founders' lineages (members alive at the end), itself
  split into fecundity (lineage size, 1 + births) and survival (the rest);
- the within-lineage term (mutation and selection among descendants);
- founder sampling (next episode's founders against this episode's
  survivors), with episodes after a predator extinction (founders from the
  recent-births archive) counted apart;

and the offspring-based differential the training logs use (a different
quantity: offspring of every animal of the episode, not lineage sizes). Per
island the terms are summed over the episode transitions (all episodes but
the last); their sum equals the change of the founders' mean from the first
to the last episode. Islands are the independent replicates: reported are
the island sums, their mean and how many islands are above zero. Also the
mean variance of the founders' speed genes.

    python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.decompose_frozen
"""

import json

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.analysis.pilot import MODULE
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_continuous_traits.decompose_frozen import OUT

TERMS = ("selection", "fecundity", "survival", "within_lineage", "sampling", "after_extinction", "logged_selection")


def island_terms(rows):
    per = {term: [] for term in TERMS}
    for this, following in zip(rows, rows[1:] + [None]):
        if this["survivors_mean"] is None:
            for term in ("selection", "fecundity", "survival", "within_lineage", "sampling"):
                per[term].append(0.0)
            per["after_extinction"].append(following["founders_mean"] - this["founders_mean"] if following else 0.0)
        else:
            per["selection"].append(this["selection"])
            per["fecundity"].append(this["fecundity"])
            per["survival"].append(this["selection"] - this["fecundity"])
            per["within_lineage"].append(this["transmission"])
            per["sampling"].append(following["founders_mean"] - this["survivors_mean"] if following else 0.0)
            per["after_extinction"].append(0.0)
        per["logged_selection"].append(this["logged_selection"] or 0.0)
    # The last episode's selection and within-lineage terms move its
    # survivors, not a later episode's founders: leave them out of the sums
    # so that the terms add up to the founders' change, first to last.
    for term in ("selection", "fecundity", "survival", "within_lineage"):
        per[term][-1] = 0.0
    return {term: np.array(values) for term, values in per.items()}


def main():
    islands = [[json.loads(line) for line in path.read_text().splitlines()] for path in sorted(OUT.glob("island_*.jsonl"))]
    terms = [island_terms(rows) for rows in islands]
    change = [rows[-1]["founders_mean"] - rows[0]["founders_mean"] for rows in islands]
    terms = [{term: values[:-1] for term, values in t.items()} for t in terms]  # transitions only
    for t, c in zip(terms, change):
        parts = t["selection"].sum() + t["within_lineage"].sum() + t["sampling"].sum() + t["after_extinction"].sum()
        if not np.isclose(parts, c):
            raise ValueError(f"terms add up to {parts}, change is {c}")
    output = {
        "islands": len(islands),
        "episodes_per_island": [len(rows) for rows in islands],
        "predator_extinctions": [sum(r["survivors_mean"] is None for r in rows) for rows in islands],
        "change_first_to_last": {"per_island": change, "mean": float(np.mean(change))},
        "transitions_per_island": [len(rows) - 1 for rows in islands],
        "founder_speed_variance": float(np.mean([np.var(r["founders"]) for rows in islands for r in rows])),
        "summed_per_island": {},
    }
    for term in TERMS:
        sums = [float(t[term].sum()) for t in terms]
        output["summed_per_island"][term] = {
            "per_island": sums,
            "mean": float(np.mean(sums)),
            "per_transition": float(np.mean(sums) / np.mean(output["transitions_per_island"])),
            "islands_above_zero": int(sum(x > 0 for x in sums)),
        }
    path = MODULE / "analysis" / "results" / "decompose_frozen.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"written {path}")
    print(f"change of founders' mean speed, first to last episode: {output['change_first_to_last']['mean']:+.3f} {np.round(change, 3)}")
    print(f"founder speed variance {output['founder_speed_variance']:.4f}")
    print(f"{'term':18s} {'summed (mean of islands)':>26s} {'per transition':>15s}  islands > 0   per island")
    for term in TERMS:
        s = output["summed_per_island"][term]
        print(f"{term:18s} {s['mean']:+26.3f} {s['per_transition']:+15.4f}  {s['islands_above_zero']}/{len(islands)}   {np.round(s['per_island'], 2)}")


if __name__ == "__main__":
    main()
