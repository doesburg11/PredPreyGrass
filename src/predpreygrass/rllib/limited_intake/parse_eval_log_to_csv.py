import json
import re

# Path to the evaluation output file (update if needed)
# Try to find evaluation_output.txt in script dir if not found in cwd
import os
EVAL_LOG = "evaluation_output.txt"
if not os.path.isfile(EVAL_LOG):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, EVAL_LOG)
    if os.path.isfile(alt_path):
        EVAL_LOG = alt_path
    else:
        raise FileNotFoundError(f"Could not find {EVAL_LOG} in current directory or script directory: {alt_path}")
# Output summary file
SUMMARY_CSV = "evaluation_step_summary.csv"

def parse_dict(line):
    """Parse a dict from a line, fallback to eval if needed."""
    try:
        # Try JSON
        return json.loads(line)
    except Exception:
        # Fallback: replace single quotes and None/True/False
        line = line.replace("'", '"')
        line = re.sub(r'\bNone\b', 'null', line)
        line = re.sub(r'\bTrue\b', 'true', line)
        line = re.sub(r'\bFalse\b', 'false', line)
        try:
            return json.loads(line)
        except Exception:
            try:
                return eval(line)
            except Exception:
                return {}

def main():
    with open(EVAL_LOG, "r") as f:
        lines = f.readlines()

    step_data = []
    current = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("step "):
            if current:
                step_data.append(current)
            current = {"step": int(line.split()[1])}
        elif line.startswith("counts predators="):
            m = re.match(r"counts predators=(\d+) prey=(\d+) agents=(\d+)", line)
            if m:
                current["num_predators"] = int(m.group(1))
                current["num_prey"] = int(m.group(2))
                current["num_agents"] = int(m.group(3))
        elif line.startswith("predators "):
            current["predators"] = parse_dict(lines[i+1].strip())
        elif line.startswith("prey "):
            current["prey"] = parse_dict(lines[i+1].strip())
        elif line.startswith("rewards "):
            current["rewards"] = parse_dict(lines[i+1].strip())
        elif line.startswith("cumulative rewards "):
            current["cumulative_rewards"] = parse_dict(lines[i+1].strip())
        elif line.startswith("terminations "):
            current["terminations"] = parse_dict(lines[i+1].strip())
        elif line.startswith("death causes "):
            current["death_causes"] = parse_dict(lines[i+1].strip())
        elif line.startswith("predators ate this step "):
            current["predators_ate"] = parse_dict(lines[i+1].strip())
    if current:
        step_data.append(current)

    # Write summary CSV
    with open(SUMMARY_CSV, "w") as f:
        header = [
            "step", "num_agents", "num_predators", "num_prey",
            "predators", "prey", "rewards", "cumulative_rewards",
            "terminations", "death_causes", "predators_ate"
        ]
        f.write(",".join(header) + "\n")
        for s in step_data:
            row = [
                str(s.get("step", "")),
                str(s.get("num_agents", "")),
                str(s.get("num_predators", "")),
                str(s.get("num_prey", "")),
                repr(s.get("predators", [])),
                repr(s.get("prey", [])),
                repr(s.get("rewards", {})),
                repr(s.get("cumulative_rewards", {})),
                repr(s.get("terminations", {})),
                repr(s.get("death_causes", {})),
                repr(s.get("predators_ate", [])),
            ]
            f.write(",".join(row) + "\n")
    print(f"Wrote step summary to {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
