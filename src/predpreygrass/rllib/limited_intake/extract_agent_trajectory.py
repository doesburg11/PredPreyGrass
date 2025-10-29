import re
import json


import os
EVAL_LOG = "evaluation_output.txt"
# Try to find evaluation_output.txt in script dir if not found in cwd
if not os.path.isfile(EVAL_LOG):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.join(script_dir, EVAL_LOG)
    if os.path.isfile(alt_path):
        EVAL_LOG = alt_path
    else:
        raise FileNotFoundError(f"Could not find {EVAL_LOG} in current directory or script directory: {alt_path}")

AGENT_ID = "type_1_predator_1"  # Change this to the agent you want to track
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"trajectory_{AGENT_ID}.csv")

def parse_dict(line):
    try:
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

    steps = []
    current_step = None
    agent_seen = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("step "):
            if current_step is not None and agent_seen:
                steps.append(current_step)
            current_step = {"step": int(line.split()[1])}
            agent_seen = False
        elif line.startswith("rewards "):
            rewards = parse_dict(lines[i+1].strip())
            print(f"Step {current_step['step']} AGENT_ID: {AGENT_ID}")
            print(f"  rewards keys: {list(rewards.keys())}")
            if AGENT_ID in rewards:
                current_step["reward"] = rewards[AGENT_ID]
                agent_seen = True
        elif line.startswith("cumulative rewards "):
            cum_rewards = parse_dict(lines[i+1].strip())
            print(f"  cumulative_rewards keys: {list(cum_rewards.keys())}")
            if AGENT_ID in cum_rewards:
                current_step["cumulative_reward"] = cum_rewards[AGENT_ID]
                agent_seen = True
        elif line.startswith("terminations "):
            terminations = parse_dict(lines[i+1].strip())
            print(f"  terminations keys: {list(terminations.keys())}")
            if AGENT_ID in terminations:
                current_step["terminated"] = terminations[AGENT_ID]
                agent_seen = True
    if current_step is not None and agent_seen:
        steps.append(current_step)
    # Debug: print how many steps found
    print(f"Extracted {len(steps)} steps for agent {AGENT_ID}")

    # Write to CSV
    with open(OUTPUT_CSV, "w") as f:
        f.write("step,reward,cumulative_reward,terminated\n")
        for s in steps:
            f.write(f"{s.get('step','')},{s.get('reward','')},{s.get('cumulative_reward','')},{s.get('terminated','')}\n")
    print(f"Wrote trajectory for {AGENT_ID} to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
