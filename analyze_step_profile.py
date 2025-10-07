import re
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python analyze_step_profile.py step_profile.log")
    sys.exit(1)

logfile = sys.argv[1]
pattern = re.compile(
    r"\\[PROFILE\\] step: trunc_check=([0-9.e-]+)s, decay=([0-9.e-]+)s, age=([0-9.e-]+)s, grass=([0-9.e-]+)s, move=([0-9.e-]+)s, engage=([0-9.e-]+)s, repro=([0-9.e-]+)s, obs=([0-9.e-]+)s, total=([0-9.e-]+)s"
)

sections = ["trunc_check", "decay", "age", "grass", "move", "engage", "repro", "obs", "total"]
data = {k: [] for k in sections}

with open(logfile) as f:
    for line in f:
        m = pattern.search(line)
        if m:
            for i, k in enumerate(sections):
                data[k].append(float(m.group(i + 1)))

print("Section      Avg (ms)   Min (ms)   Max (ms)")
for k in sections:
    if data[k]:
        avg = 1000 * sum(data[k]) / len(data[k])
        mn = 1000 * min(data[k])
        mx = 1000 * max(data[k])
        print(f"{k:10} {avg:9.3f} {mn:9.3f} {mx:9.3f}")

# Plot
plt.figure(figsize=(10, 6))
for k in sections:
    if k != "total":
        plt.plot(data[k], label=k)
plt.plot(data["total"], label="total", linewidth=2, color="black")
plt.xlabel("Step")
plt.ylabel("Time (s)")
plt.title("Step Timing Breakdown")
plt.legend()
plt.tight_layout()
plt.show()
