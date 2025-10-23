import io
import contextlib
from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

def run_and_capture_logs():
    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream):
        env = PredPreyGrassEnv(initial_num_predators=3, initial_num_prey=2, num_possible_predators=3, num_possible_prey=2)
        env.reset()
        # Set all agents' energy to a small value so they will starve
        env.energy[:] = 0.01
        # Step with no actions
        env.step({})
    return log_stream.getvalue()

def main():
    logs = run_and_capture_logs()
    print("Captured logs:\n" + logs)
    # Check for double-logging
    lines = logs.splitlines()
    starved = set()
    engaged = set()
    for line in lines:
        if line.startswith('[Starvation] Deactivated agents:'):
            starved.update(eval(line.split(':', 1)[1].strip()))
        if line.startswith('[Engagement] Deactivated agents:'):
            engaged.update(eval(line.split(':', 1)[1].strip()))
    double_logged = starved & engaged
    if double_logged:
        print(f"ERROR: Agents double-logged as deactivated by both starvation and engagement: {sorted(double_logged)}")
    else:
        print("PASS: No agents were double-logged as deactivated.")

if __name__ == "__main__":
    main()
