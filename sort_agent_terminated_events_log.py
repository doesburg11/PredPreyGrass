# Sort agent_terminated_events.log by episode number
import re

input_path = "src/predpreygrass/rllib/limited_intake/logs/agent_terminated_events.log"
output_path = input_path  # Overwrite in place

with open(input_path, "r") as f:
    lines = f.readlines()


def extract_episode_and_agent(line):
    m = re.search(r"episode=(\d+)", line)
    episode = int(m.group(1)) if m else float('inf')
    m2 = re.search(r"agent_id=([^ ]+)", line)
    agent = m2.group(1) if m2 else "zzzzzzzz"
    return (episode, agent)

lines_sorted = sorted(lines, key=extract_episode_and_agent)

with open(output_path, "w") as f:
    f.writelines(lines_sorted)
