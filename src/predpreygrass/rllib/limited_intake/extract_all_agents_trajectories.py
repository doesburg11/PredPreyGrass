import re
import json

EVAL_LOG = "evaluation_output.txt"
import os
if not os.path.isfile(EVAL_LOG):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, EVAL_LOG)
    if os.path.isfile(alt_path):
        EVAL_LOG = alt_path
    else:
        raise FileNotFoundError(f"Could not find {EVAL_LOG} in current directory or script directory: {alt_path}")

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_agents_trajectories.csv")

def parse_dict(line):
    try:
        return json.loads(line)
    except Exception:
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
                print(f"Failed to parse dict: {line}")
                return {}

def main():
    with open(EVAL_LOG, "r") as f:
        lines = f.readlines()

    step = None
    rewards = {}
    cum_rewards = {}
    terminations = {}
    all_rows = []
    def flush_step(step, rewards, cum_rewards, terminations):
        agent_ids = set(rewards.keys()) | set(cum_rewards.keys()) | set(terminations.keys())
        print(f"Flushing step {step}: {len(agent_ids)} agents")
        print(f"  rewards: {list(rewards.keys())[:3]}... ({len(rewards)})")
        print(f"  cum_rewards: {list(cum_rewards.keys())[:3]}... ({len(cum_rewards)})")
        print(f"  terminations: {list(terminations.keys())[:3]}... ({len(terminations)})")
        for aid in agent_ids:
            row = {
                "step": step,
                "agent_id": aid,
                "reward": rewards.get(aid, ""),
                "cumulative_reward": cum_rewards.get(aid, ""),
                "terminated": terminations.get(aid, "")
            }
            all_rows.append(row)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("step "):
            print(f"Found {line}")
            if step is not None:
                flush_step(step, rewards, cum_rewards, terminations)
            step = int(line.split()[1])
            rewards = {}
            cum_rewards = {}
            terminations = {}
            i += 1
            continue
        elif line.startswith("rewards"):
            print(f"  Found rewards at line {i}")
            # Find next non-empty, non-separator line
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("-"):
                    print(f"    rewards dict: {next_line[:60]}...")
                    rewards = parse_dict(next_line)
                    break
                j += 1
            i = j
            continue
        elif line.startswith("cumulative rewards"):
            print(f"  Found cumulative rewards at line {i}")
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("-"):
                    print(f"    cum_rewards dict: {next_line[:60]}...")
                    cum_rewards = parse_dict(next_line)
                    break
                j += 1
            i = j
            continue
        elif line.startswith("terminations"):
            print(f"  Found terminations at line {i}")
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("-"):
                    print(f"    terminations dict: {next_line[:60]}...")
                    terminations = parse_dict(next_line)
                    break
                j += 1
            i = j
            continue
        i += 1
    # Flush last step
    if step is not None:
        flush_step(step, rewards, cum_rewards, terminations)

    # Write to CSV
    with open(OUTPUT_CSV, "w") as f:
        f.write("step,agent_id,reward,cumulative_reward,terminated\n")
        for row in all_rows:
            f.write(f"{row['step']},{row['agent_id']},{row['reward']},{row['cumulative_reward']},{row['terminated']}\n")
    print(f"Wrote all agent trajectories to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
