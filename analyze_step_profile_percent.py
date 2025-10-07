
import re
import sys

if len(sys.argv) < 2:
    logfile = "step_profile.log"
    print(f"No log file argument provided, defaulting to {logfile}")
else:
    logfile = sys.argv[1]

# Remove ANSI color codes
ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

# More flexible pattern (allow spaces, color codes, etc.)
pattern = re.compile(
    r"\[PROFILE\] step: trunc_check=([0-9.e-]+)s, decay=([0-9.e-]+)s, age=([0-9.e-]+)s, grass=([0-9.e-]+)s, move=([0-9.e-]+)s, engage=([0-9.e-]+)s, repro=([0-9.e-]+)s, obs=([0-9.e-]+)s, total=([0-9.e-]+)s"
)

sections = ["trunc_check", "decay", "age", "grass", "move", "engage", "repro", "obs", "total"]
data = {k: [] for k in sections}

with open(logfile) as f:
    for line in f:
        # Remove ANSI color codes
        line = ansi_escape.sub('', line)
        m = pattern.search(line)
        if m:
            for i, k in enumerate(sections):
                data[k].append(float(m.group(i + 1)))

section_sums = {k: sum(data[k]) for k in sections if k != "total"}
total_sum = sum(data["total"])

print("Section      Avg ms   Avg % of step time")
if total_sum == 0:
    print("No matching [PROFILE] step: lines found. Check log format or pattern.")
else:
    n = len(data["total"])
    engage_ms = [1000 * v for v in data["engage"]]
    avg_engage_ms = sum(engage_ms) / n if n > 0 else 0
    max_engage_ms = max(engage_ms) if engage_ms else 0
    for k in sections:
        if k != "total":
            avg_ms = 1000 * sum(data[k]) / n if n > 0 else 0
            percent = 100 * section_sums[k] / total_sum if total_sum > 0 else 0
            print(f"{k:10} {avg_ms:7.3f} ms   {percent:10.2f}%")
    # Print total row
    avg_total_ms = 1000 * sum(data["total"]) / n if n > 0 else 0
    print(f"{'total':10} {avg_total_ms:7.3f} ms   {100.00:10.2f}%")
    print(f"\n[engage] max: {max_engage_ms:.3f} ms, avg: {avg_engage_ms:.3f} ms")
    # Flag spikes > 2x average
    spike_indices = [i for i, v in enumerate(engage_ms) if v > 2 * avg_engage_ms]
    if spike_indices:
        print(f"[engage] Spikes (>2x avg) at steps: {spike_indices[:10]}{'...' if len(spike_indices)>10 else ''} (total {len(spike_indices)})")
    else:
        print("[engage] No spikes (>2x avg) detected.")
