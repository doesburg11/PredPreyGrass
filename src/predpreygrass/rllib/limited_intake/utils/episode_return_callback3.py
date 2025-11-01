from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict
import os

class EpisodeReturn(RLlibCallback):
    def __init__(self, *, lookback_k: int = 5):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.episode_lengths = {}  # manual episode length tracking
        self._episode_los_rejected = {}
        self.lookback_k = lookback_k  # NEW: how many env steps to show around an event
        # --- Human-readable episode numbering ---
        self.episode_counter = 0
        self.episode_id_to_number = {}

    def on_episode_start(self, *, episode, **kwargs):
        eid = episode.id_
        if eid not in self.episode_id_to_number:
            self.episode_counter += 1
            self.episode_id_to_number[eid] = self.episode_counter

    # ---------- Helpers ----------
    def _episode_agent_ids(self, episode) -> list:
        """
        Return a list of agent IDs for the current episode in a way that's
        compatible with RLlib's evolving Episode APIs.
        """
        # Try common stable attributes/methods first
        for attr in ("agent_ids", "get_agent_ids", "get_agents"):
            if hasattr(episode, attr):
                obj = getattr(episode, attr)
                return list(obj() if callable(obj) else obj)
        if hasattr(episode, "last_info_for") and hasattr(episode, "_agent_to_last_info"):
            return list(getattr(episode, "_agent_to_last_info").keys())
        return []

    def _episode_last_infos(self, episode) -> dict:
        for name in ("get_last_infos", "get_infos"):
            if hasattr(episode, name):
                infos = getattr(episode, name)()
                if isinstance(infos, dict):
                    return infos
        for name in ("last_infos", "infos"):
            if hasattr(episode, name):
                infos = getattr(episode, name)
                if isinstance(infos, dict):
                    return infos
        if hasattr(episode, "_agent_to_last_info"):
            mapping = getattr(episode, "_agent_to_last_info")
            if isinstance(mapping, dict):
                return mapping
        return {}

    def _env_len(self, episode) -> int:
        # Robust way to infer env length: take max of reward sequences
        rewards_map = episode.get_rewards()
        return max((len(seq) for seq in rewards_map.values()), default=0)

    def _print_window_for_agents(self, *, episode, agents, title: str):
        # Debug: Print what agent IDs and data are actually present in the episode object
        all_rewards = episode.get_rewards()
        print(f"[DEBUG] episode.get_rewards() keys: {list(all_rewards.keys())}")
        all_actions = episode.get_actions() if hasattr(episode, 'get_actions') else None
        if all_actions is not None:
            print(f"[DEBUG] episode.get_actions() keys: {list(all_actions.keys())}")
        all_terms = episode.get_terminateds() if hasattr(episode, 'get_terminateds') else None
        if all_terms is not None:
            print(f"[DEBUG] episode.get_terminateds() keys: {list(all_terms.keys())}")
        env_len = self._env_len(episode)
        if env_len == 0:
            return
        start = max(0, env_len - self.lookback_k)
        end = env_len
        """
        print(f"\n=== EVENT WINDOW ({title}) env_t in [{start}, {end}) ===")
        # Which agents stepped each env step?
        for t in range(start, end):
            try:
                stepped = episode.get_agents_that_stepped(env_t=t)
            except Exception:
                stepped = None
            if stepped:
                print(f"  env_t={t}: stepped={stepped}")

        # Per-agent actions/rewards/term flags
        for aid in agents:
            # Try windowed access first
            try:
                acts = episode.get_actions(agent_id=aid, start=start, end=end)
            except Exception:
                acts = None
            try:
                rws = episode.get_rewards(agent_id=aid, start=start, end=end)
            except Exception:
                rws = None
            try:
                terms = episode.get_terminateds(agent_id=aid, start=start, end=end)
            except Exception:
                terms = None

            # If all are None, fallback to full lists
            if acts is None:
                try:
                    acts = all_actions[aid] if all_actions and aid in all_actions else None
                except Exception:
                    acts = None
            if rws is None:
                try:
                    rws = all_rewards[aid] if all_rewards and aid in all_rewards else None
                except Exception:
                    rws = None
            if terms is None:
                try:
                    terms = all_terms[aid] if all_terms and aid in all_terms else None
                except Exception:
                    terms = None

            if acts is not None or rws is not None or terms is not None:
                print(f"  Agent {aid}:")
                if acts is not None:
                    print(f"    actions={acts}")
                if rws is not None:
                    print(f"    rewards={rws}")
                if terms is not None:
                    print(f"    terminateds={terms}")
        """
    # ---------- RLlib hooks ----------
    def on_episode_step(self, *, episode, **kwargs):
        eid = episode.id_
        self.episode_lengths[eid] = self.episode_lengths.get(eid, 0) + 1
        step_num = self.episode_lengths[eid]
        # Assign a simple incrementing episode number (1, 2, 3, ...)
        if eid not in self.episode_id_to_number:
            self.episode_counter += 1
            self.episode_id_to_number[eid] = self.episode_counter
        ep_num = self.episode_id_to_number[eid]
        # Write step number with a unique marker to a dedicated log file
        log_path = os.path.join(os.path.dirname(__file__), "../logs/step_numbers.log")
        with open(log_path, "a") as f:
            f.write(f"[RL_STEP_NUM] STEP {step_num} (Episode {ep_num})\n")

        # --- NEW: Log per-agent termination events ---
        terminateds = None
        if hasattr(episode, 'get_terminateds'):
            terminateds = episode.get_terminateds()
        elif hasattr(episode, '_agent_done'):
            terminateds = getattr(episode, '_agent_done')
        if terminateds and isinstance(terminateds, dict):
            for agent_id, term_val in terminateds.items():
                # RLlib may use True/False or 1/0
                if term_val is True or term_val == 1:
                    log_path = os.path.join(os.path.dirname(__file__), "../logs/agent_terminated_events.log")
                    with open(log_path, "a") as tf:
                        tf.write(f"[AGENT_TERMINATED] agent_id={agent_id} episode={ep_num} step={step_num}\n")
        # Aggregate per-agent infos for LOS rejections into an episode counter
        los_count = 0
        infos_map = self._episode_last_infos(episode)
        if isinstance(infos_map, dict):
            for info in infos_map.values():
                if info and isinstance(info, dict):
                    los_count += int(info.get("los_rejected", 0))
        self._episode_los_rejected[eid] = self._episode_los_rejected.get(eid, 0) + los_count

        # ---- NEW: Detect interesting events from infos and print a short window
        interesting_sets = {
            "predator_ate_prey": set(),
            "prey_ate_grass": set(),
            "caught_prey": set(),
            "spawned": set(),
        }
        for agent_id, info in (infos_map or {}).items():
            if not info:
                continue
            if "ate_prey" in info:
                interesting_sets["predator_ate_prey"].add(agent_id)
                # also include the prey id to inspect its termination/reward
                interesting_sets["caught_prey"].add(info["ate_prey"])
            if "got_caught_by" in info:
                interesting_sets["caught_prey"].add(agent_id)
                interesting_sets["predator_ate_prey"].add(info["got_caught_by"])
            if "ate_grass" in info:
                interesting_sets["prey_ate_grass"].add(agent_id)
            if "spawned" in info:
                interesting_sets["spawned"].update([agent_id, info["spawned"]])
            if "born_of" in info:
                interesting_sets["spawned"].update([agent_id, info["born_of"]])

        # Always print the window for all agents at every step
        all_agents = self._episode_agent_ids(episode)
        if all_agents:
            self._print_window_for_agents(episode=episode, agents=all_agents, title="all agents (every step)")
        
        # Print terminateds (done flags) for each agent if available
        # RLlib episode object may have _agent_done or _agent_to_last_done_step
        if hasattr(episode, '_agent_to_last_done_step'):
            print("[Callback] Terminateds (last done step per agent):")
            for agent_id, done_step in episode._agent_to_last_done_step.items():
                print(f"  {agent_id}: last done at step {done_step}")
        elif hasattr(episode, '_agent_done'):
            print("[Callback] Terminateds (_agent_done):")
            for agent_id, done in episode._agent_done.items():
                print(f"  {agent_id}: {done}")
        else:
            print("[Callback] No per-agent terminateds found in episode object.")

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
        # Assign a human-readable episode number if not already assigned
        eid = episode.id_
        if eid not in self.episode_id_to_number:
            self.episode_counter += 1
            self.episode_id_to_number[eid] = self.episode_counter
        """
        Called at the end of each episode.
        Logs the total and average rewards separately for predators and prey.
        """
        self.num_episodes += 1
        episode_return = episode.get_return()
        episode_id = episode.id_
        episode_length = self.episode_lengths.pop(episode_id, 0)
        self.overall_sum_of_rewards += episode_return

        # Accumulate rewards by group
        group_rewards = defaultdict(list)
        predator_total = prey_total = 0.0

        for agent_id, rewards in episode.get_rewards().items():
            total = sum(rewards)
            if "predator" in agent_id:
                predator_total += total
            elif "prey" in agent_id:
                prey_total += total

            for group in ["type_1_predator", "type_2_predator", "type_1_prey", "type_2_prey"]:
                if group in agent_id:
                    group_rewards[group].append(total)
                    break

        print(
            f"Episode {self.num_episodes}: Length: {episode_length} | R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}"
        )
        print(f"  - Predators: Total = {predator_total:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}")
        for group, totals in group_rewards.items():
            print(f"  - {group}: Total = {sum(totals):.2f}")

        metrics_logger.log_value("episode_length", episode_length, reduce="mean")
        los_rejected = self._episode_los_rejected.pop(episode_id, 0)
        metrics_logger.log_value("los_rejected_moves", los_rejected, reduce="mean")

    def on_train_result(self, *, result, **kwargs):
        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0
