"""RESULTS.md section 3.5 (exploratory): arm B with the innate bias off at
evaluation (launchers/bias_off.py) against bias on (section 3.3's results).

Premise: a B run meets it at a checkpoint if its own-world evaluation (bias
on, 50 episodes) has mean a_young > 0 and mean a_adult < 0 (the episodes'
gene means). P1 and P2 are interpreted only if at least 4 of 6 B runs meet
it at each of 500, 750 and 1000.

P1 (within B, paired, own world): per B run, bias off minus bias on of A
(adults per episode) and of C (coop_fair adult share over chance), each the
mean over the three checkpoints; one-sided sign-flip permutation test over
the 64 sign arrangements, p = share of arrangements whose mean is >= the
observed one; p < 0.05 for each measure.

P2 (common panel): B's panel values with the bias off (33 off-diagonal
cells per run) against L's panel values (section 3.4), A and C, section
3.4's 924-labelling permutation test, one-sided B-off > L, p < 0.05 each.

"consistent with H_nn" only if P1 holds on both A and C; P2 reported
separately. Also reported: S, young caught solo and with a partner, adult
share of catches, hunter births, mean hunter and deer numbers, in both
conditions, and P1's per-checkpoint differences.

Run from the repository root:
    python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/verdict_3_5.py
"""

import itertools
import json

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.bias_off import OUT
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.verdict_3 import CHECKPOINTS, RESULTS, RUNS, check_own, load_panel, panel_values, permutation_p

ALPHA = 0.05
PREMISE_RUNS = 4


def sign_flip_p(diffs):
    """(mean difference, one-sided p over all 2^n sign arrangements)."""
    n = len(diffs)
    observed = sum(diffs) / n
    count = total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        count += sum(s * d for s, d in zip(signs, diffs)) / n >= observed - 1e-12
    return observed, count / total


def measures(summary, coop):
    """Per-condition values of one evaluation: A, C and the reported ones."""
    t, n = summary["totals"], summary["episodes"]
    caught = t.get("adults_caught", 0)
    return {
        "A": caught / n,
        "C": coop["adult_over_chance"],
        "caught": caught,
        "failures": t.get("adult_failures", 0),
        "young_solo": t.get("young_caught_solo", 0) / n,
        "young_party": t.get("young_caught_party", 0) / n,
        "adult_catch_share": summary.get("adult_catch_share"),
        "births": t.get("births_predator", 0) / n,
        "hunters": t.get("predator_steps", 0) / max(t.get("physics_steps", 0), 1),
        "deer": t.get("deer_steps", 0) / max(t.get("physics_steps", 0), 1),
        "coexisting": summary["coexisting"],
    }


def check_off(data, hunter_run, deer_run, mark):
    """Refuse a bias-off file that is not this hunter checkpoint against this
    deer run's checkpoint deer at this mark (provenance recomputed from the
    module files), with beta 0, 50 evaluation episodes with seeds 0-49 and
    20 coop_fair episodes."""
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_at, checkpoint_deer
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_ppo import RUNS_DIR
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.launchers.cross_play import provenance

    label = f"bias-off {hunter_run} vs {deer_run} at {mark}"
    expected = dict(provenance(checkpoint_at(RUNS_DIR / hunter_run, mark), checkpoint_deer(checkpoint_at(RUNS_DIR / deer_run, mark))), beta_at_evaluation=0.0)
    if data.get("provenance") != expected or (data.get("hunter_run"), data.get("deer_run"), data.get("mark")) != (hunter_run, deer_run, mark):
        raise ValueError(f"{label}: provenance differs")
    if data["summary"].get("beta_at_evaluation") != 0.0:
        raise ValueError(f"{label}: not evaluated with beta 0")
    if data["summary"]["episodes"] != 50 or sorted(row["seed"] for row in data["episodes"]) != list(range(50)):
        raise ValueError(f"{label}: not 50 episodes with seeds 0-49")
    if data["coop"]["episodes"] != 20 or len(data.get("coop_episodes") or []) != 20:
        raise ValueError(f"{label}: not 20 coop_fair episodes")


def own_on(run, mark):
    s = json.loads((RESULTS / f"{run}_iter{mark}_eval.json").read_text())["summary"]
    c = json.loads((RESULTS / f"{run}_iter{mark}_coop.json").read_text())["summary"]
    return s, c


def own_off(run, mark):
    data = json.loads((OUT / f"{run}_iter{mark}_eval.json").read_text())
    check_off(data, run, run, mark)
    return data["summary"], data["coop"]


def check_same_episodes(on, off, label):
    if on["episodes"] != off["episodes"] or on["checkpoint"] != off["checkpoint"] or on["deer"] != off["deer"]:
        raise ValueError(f"{label}: bias-on and bias-off evaluations differ in checkpoint, deer or episodes")


def run_mean(per, key):
    values = [p[key] for p in per if p[key] is not None]
    return sum(values) / len(values) if values else None


def load_off_panel():
    """{(hunters, deer, mark): summary with coop} for B's bias-off cells."""
    panel = {}
    everyone = RUNS["L"] + RUNS["B"]
    for mark in CHECKPOINTS:
        for h in RUNS["B"]:
            for d in everyone:
                if d == h:
                    continue
                data = json.loads((OUT / f"{h}_iter{mark}_vs_{d}_eval.json").read_text())
                check_off(data, h, d, mark)
                panel[(h, d, mark)] = dict(data["summary"], coop=data["coop"])
    return panel


def main():
    for run in RUNS["L"] + RUNS["B"]:
        for mark in CHECKPOINTS:
            check_own(run, mark)
    premise, statuses = {}, []
    for mark in CHECKPOINTS:
        met = []
        for run in RUNS["B"]:
            genes = own_on(run, mark)[0]["genes"]
            ok = genes["gene_a_young_mean"] > 0 and genes["gene_a_adult_mean"] < 0
            met.append(ok)
            statuses.append(f"  {run} iter {mark}: a_young {genes['gene_a_young_mean']:+.3f}, a_adult {genes['gene_a_adult_mean']:+.3f} -> {'meets' if ok else 'does not meet'}")
        premise[mark] = sum(met)
    interpret = all(v >= PREMISE_RUNS for v in premise.values())

    per_run = {}
    for run in RUNS["B"]:
        rows = []
        for mark in CHECKPOINTS:
            (s_on, c_on), (s_off, c_off) = own_on(run, mark), own_off(run, mark)
            check_same_episodes(s_on, s_off, f"{run} iter {mark}")
            rows.append((mark, measures(s_on, c_on), measures(s_off, c_off)))
        per_run[run] = rows

    p1 = {}
    for key in ("A", "C"):
        diffs = []
        for run in RUNS["B"]:
            on = run_mean([r[1] for r in per_run[run]], key)
            off = run_mean([r[2] for r in per_run[run]], key)
            diffs.append(off - on)
        mean, p = sign_flip_p(diffs)
        p1[key] = {"differences": diffs, "mean": mean, "p": p, "met": p < ALPHA}

    off_panel = load_off_panel()
    l_panel = load_panel(RUNS["L"] + RUNS["B"])
    b_off = [panel_values(off_panel, r) for r in RUNS["B"]]
    l_vals = [panel_values(l_panel, r) for r in RUNS["L"]]
    b_on = [panel_values(l_panel, r) for r in RUNS["B"]]
    p2 = {}
    for key in ("A", "C"):
        diff, p = permutation_p([v[key] for v in b_off], [v[key] for v in l_vals])
        p2[key] = {"difference": diff, "p": p, "met": p < ALPHA}

    print("premise (a_young > 0 and a_adult < 0, bias on): " + ", ".join(f"{m}: {premise[m]} of 6" for m in CHECKPOINTS) + f" -> {'interpreted' if interpret else 'reported only, not interpreted'}")
    print("\n".join(statuses))

    def label(met):
        return f" ({'met' if met else 'not met'})" if interpret else ""

    for key in ("A", "C"):
        d = p1[key]
        print(f"P1 {key} (bias off - bias on, own world, mean of 500/750/1000): " + "; ".join(f"{r} {x:+.3f}" for r, x in zip(RUNS["B"], d["differences"])) + f"; mean {d['mean']:+.4f}, sign-flip p {d['p']:.4f}{label(d['met'])}")
    for key in ("A", "C"):
        d = p2[key]
        print(f"P2 {key} (B bias off - L, panel): {d['difference']:+.4f}, permutation p {d['p']:.4f}{label(d['met'])}")
    print("panel values per B run (bias on -> off), and L:")
    for r, on, off in zip(RUNS["B"], b_on, b_off):
        print(f"  {r}: A {on['A']:.2f} -> {off['A']:.2f}; S {on['S']:.4f} -> {off['S']:.4f}; C {on['C']:.3f} -> {off['C']:.3f}")
    for r, v in zip(RUNS["L"], l_vals):
        print(f"  {r}: A {v['A']:.2f}; S {v['S']:.4f}; C {v['C']:.3f}")
    print("own world per checkpoint (bias on -> off): A, C, S, young solo, young with partner, adult share of catches, births, mean hunters and deer (time-weighted), coexisting")
    for run in RUNS["B"]:
        for mark, on, off in per_run[run]:
            def s(m):
                attempts = m["caught"] + m["failures"]
                return m["caught"] / attempts if attempts else 0.0

            def f(key, digits=2):
                a, b = on[key], off[key]
                return f"{a:.{digits}f} -> {b:.{digits}f}" if a is not None and b is not None else f"{a} -> {b}"

            dA = off["A"] - on["A"]
            dC = off["C"] - on["C"] if on["C"] is not None and off["C"] is not None else None
            print(f"  {run} iter {mark}: A {f('A')} (off - on {dA:+.2f}); C {f('C', 3)} (off - on {dC if dC is None else round(dC, 3)}); S {s(on):.4f} -> {s(off):.4f}; young solo {f('young_solo')}; young party {f('young_party')}; adult share {f('adult_catch_share', 3)}; births {f('births')}; mean hunters {f('hunters', 1)}; mean deer {f('deer', 1)}; coexisting {on['coexisting']} -> {off['coexisting']}")
    consistent = p1["A"]["met"] and p1["C"]["met"]
    if interpret:
        print("H_nn: consistent (P1 met on A and C)" if consistent else "H_nn: not supported (P1 not met on both A and C)")
    else:
        print("H_nn: premise not met; P1 and P2 reported, not interpreted")


if __name__ == "__main__":
    main()
