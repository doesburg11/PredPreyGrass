from collections import defaultdict
import math

class ContextualCaptureTracker:
    def __init__(self, max_window_steps=5):
        self.max_window_steps = max_window_steps
        self.visibility_log = defaultdict(list)  # (pred_id) -> list of (prey_id, step_seen)
        self.captures = defaultdict(int)        # (pred_id) -> contextual captures
        self.total_visible = defaultdict(int)   # (pred_id) -> total prey seen

    def log_visibility(self, predator_id, visible_prey_ids, current_step):
        for prey_id in visible_prey_ids:
            self.visibility_log[predator_id].append((prey_id, current_step))
            self.total_visible[predator_id] += 1

    def log_capture(self, predator_id, prey_id, current_step):
        # Check if this prey was seen in the last N steps by this predator
        for prey_seen, step_seen in reversed(self.visibility_log[predator_id]):
            if prey_seen == prey_id and (current_step - step_seen) <= self.max_window_steps:
                self.captures[predator_id] += 1
                break  # Count only once per capture

    def summarize_by_speed(self):
        summary = {"speed_1": {"captures": 0, "seen": 0}, "speed_2": {"captures": 0, "seen": 0}}
        for pid in self.total_visible:
            speed = "speed_1" if "speed_1" in pid else "speed_2"
            summary[speed]["captures"] += self.captures.get(pid, 0)
            summary[speed]["seen"] += self.total_visible[pid]
        return summary

    def print_summary(self):
        summary = self.summarize_by_speed()
        for speed, stats in summary.items():
            seen = stats["seen"]
            captured = stats["captures"]
            efficiency = captured / seen if seen > 0 else 0
            print(f"{speed.upper()} Contextual Capture Efficiency: {captured}/{seen} = {efficiency:.2%}")

class FocusedPursuitTracker:
    def __init__(self, max_window_steps=5):
        self.max_window_steps = max_window_steps
        self.attempts = {}  # predator_id -> (target_prey_id, step_seen)
        self.successes = defaultdict(int)  # predator_id -> count
        self.total_attempts = defaultdict(int)

    def log_focus(self, predator_id, predator_pos, visible_prey_positions, current_step):
        if not visible_prey_positions:
            return

        # Choose the closest prey (simple heuristic for intent)
        closest_prey_id = min(
            visible_prey_positions,
            key=lambda pid: self._distance(predator_pos, visible_prey_positions[pid])
        )
        self.attempts[predator_id] = (closest_prey_id, current_step)
        self.total_attempts[predator_id] += 1

    def log_capture(self, predator_id, prey_id, current_step):
        if predator_id not in self.attempts:
            return

        target_id, step_seen = self.attempts[predator_id]
        if prey_id == target_id and (current_step - step_seen) <= self.max_window_steps:
            self.successes[predator_id] += 1

    def summarize_by_speed(self):
        summary = {"speed_1": {"captures": 0, "attempts": 0}, "speed_2": {"captures": 0, "attempts": 0}}
        for pid in self.total_attempts:
            speed = "speed_1" if "speed_1" in pid else "speed_2"
            summary[speed]["captures"] += self.successes.get(pid, 0)
            summary[speed]["attempts"] += self.total_attempts[pid]
        return summary

    def print_summary(self):
        summary = self.summarize_by_speed()
        for speed, stats in summary.items():
            attempts = stats["attempts"]
            captures = stats["captures"]
            efficiency = captures / attempts if attempts > 0 else 0
            print(f"{speed.upper()} Focused Pursuit Efficiency: {captures}/{attempts} = {efficiency:.2%}")

    @staticmethod
    def _distance(pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
