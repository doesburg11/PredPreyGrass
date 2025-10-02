from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
import time
from collections import defaultdict


class EpisodeReturn(RLlibCallback):
    def on_episode_step(self, *, episode, **kwargs):
        # ...existing code...
        # Windowed lineage reward hook (Tier-1 Selfish Gene)
        infos_map = self._episode_last_infos(episode)
        window = 50  # can be made configurable
        for agent_id in self._episode_agent_ids(episode):
            env = getattr(episode, "env", None)
            if env and hasattr(env, "_windowed_lineage_reward"):
                lineage_reward = env._windowed_lineage_reward(agent_id, window=window)
                if agent_id not in infos_map:
                    infos_map[agent_id] = {}
                infos_map[agent_id]["windowed_lineage_reward"] = lineage_reward
        # ...existing code...
        # --- Online cooperation metrics accumulation (lightweight proxies) ---
        # We accumulate per-episode stats and emit them in on_episode_end to TensorBoard.
        try:
            env = getattr(episode, "env", None)
            if env is None:
                return
            ep_id = getattr(episode, "id_", None)
            if ep_id is None:
                return
            # Initialize per-episode accumulators if needed
            acc = self._coop_accumulators.setdefault(ep_id, {"ai_sum_raw": 0.0, "ai_sum_los": 0.0, "ai_n": 0})
            kpa = self._kpa_state.setdefault(ep_id, {
                "prev_offspring": {},  # agent_id -> int
                "prev_kin": {},        # agent_id -> bool
                "with_trials": 0,
                "with_events": 0,
                "without_trials": 0,
                "without_events": 0,
            })

            # Active learning agents this step
            agents = [a for a in getattr(env, "agents", []) if a in env.agent_positions]
            if not agents:
                return

            # Prepare lineage roots per agent and positions
            R = int(getattr(env, "kin_density_radius", 2))
            los_aware = bool(getattr(env, "kin_density_los_aware", False))
            positions = {a: env.agent_positions[a] for a in agents}
            roots = {}
            for a in agents:
                uid = env.unique_agents.get(a)
                st = env.unique_agent_stats.get(uid, {}) if uid else {}
                roots[a] = st.get("root_ancestor")

            # Compute per-agent neighbor totals and same-root counts
            def same_root_counts(use_los: bool):
                same_total = []  # list of (same, total) per agent when total>0
                kin_present = {}  # agent_id -> bool
                for a in agents:
                    ax, ay = positions[a]
                    ra = roots.get(a)
                    total = 0
                    same = 0
                    any_kin = False
                    for b in agents:
                        if b == a:
                            continue
                        bx, by = positions[b]
                        if max(abs(bx - ax), abs(by - ay)) <= R:
                            if (not use_los) or env._line_of_sight_clear((ax, ay), (bx, by)):
                                total += 1
                                if roots.get(b) is not None and roots.get(b) == ra:
                                    same += 1
                                    any_kin = True
                    if total > 0:
                        same_total.append((same, total))
                    kin_present[a] = any_kin
                return same_total, kin_present

            same_total_raw, kin_present_raw = same_root_counts(use_los=False)
            # Accumulate AI raw
            for same, total in same_total_raw:
                acc["ai_sum_raw"] += (same / total)
                acc["ai_n"] += 1

            # Optionally also accumulate LOS-aware AI (if enabled in env)
            if los_aware:
                same_total_los, _ = same_root_counts(use_los=True)
                for same, total in same_total_los:
                    acc["ai_sum_los"] += (same / total)

            # KPA proxy: use last step's offspring counts and kin-present flag to see if reproduction happened
            # Get current offspring counts from env.per_step_agent_data[-1]
            if getattr(env, "per_step_agent_data", None):
                cur = env.per_step_agent_data[-1]
                for a in agents:
                    off_now = int(cur.get(a, {}).get("offspring_count", 0))
                    if a in kpa["prev_offspring"]:
                        reproduced = off_now > kpa["prev_offspring"][a]
                        if kpa["prev_kin"].get(a, False):
                            kpa["with_trials"] += 1
                            if reproduced:
                                kpa["with_events"] += 1
                        else:
                            kpa["without_trials"] += 1
                            if reproduced:
                                kpa["without_events"] += 1
                    # Update prev state for next step
                    kpa["prev_offspring"][a] = off_now
                    kpa["prev_kin"][a] = bool(kin_present_raw.get(a, False))
        except Exception:
            # Be robust to any env/callback API mismatches; skip metrics if unavailable
            pass
    def __init__(self):
        super().__init__()
        self.overall_sum_of_rewards = 0.0
        self.num_episodes = 0
        self._pending_episode_metrics = []
        self.start_time = time.time()
        self.last_iteration_time = self.start_time
        self.episode_lengths = {}  # manual episode length tracking
        self._episode_los_rejected = {}
        # Online cooperation metrics accumulators (per episode id)
        self._coop_accumulators = {}
        self._kpa_state = {}

    def _episode_agent_ids(self, episode) -> list:
        """
        Return a list of agent IDs for the current episode in a way that's
        compatible with RLlib's evolving Episode APIs.
        """
        # Try common stable attributes/methods first
        for attr in ("agent_ids", "get_agent_ids", "get_agents"):
            if hasattr(episode, attr):
                obj = getattr(episode, attr)
                try:
                    return list(obj() if callable(obj) else obj)
                except Exception:
                    pass
        # Last resort: peek into known internal mapping used by last_info_for
        if hasattr(episode, "last_info_for") and hasattr(episode, "_agent_to_last_info"):
            try:
                return list(getattr(episode, "_agent_to_last_info").keys())
            except Exception:
                pass
        return []

    # ...existing code...

    def on_episode_end(self, *, episode, metrics_logger: MetricsLogger, **kwargs):
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

            # Match subgroup
            for group in ["type_1_predator", "type_2_predator", "type_1_prey", "type_2_prey"]:
                if group in agent_id:
                    group_rewards[group].append(total)
                    break

        # --- Lineage reward logging ---
        infos_map = self._episode_last_infos(episode)
        lineage_rewards = {}
        for agent_id in self._episode_agent_ids(episode):
            info = infos_map.get(agent_id, {})
            # Defensive: info may be a list or other type
            lineage_reward = None
            if isinstance(info, dict):
                lineage_reward = info.get("windowed_lineage_reward", None)
            elif isinstance(info, list):
                # Try to find a dict in the list
                for entry in info:
                    if isinstance(entry, dict) and "windowed_lineage_reward" in entry:
                        lineage_reward = entry["windowed_lineage_reward"]
                        break
            if lineage_reward is not None:
                lineage_rewards[agent_id] = lineage_reward
                metrics_logger.log_value(f"lineage_reward/{agent_id}", lineage_reward, reduce="mean")

        if lineage_rewards:
            print("  - Lineage rewards (windowed):")
            for agent_id, reward in lineage_rewards.items():
                print(f"    {agent_id}: {reward}")

        # Episode summary log
        print(
            f"Episode {self.num_episodes}: Length: {episode_length} | R={episode_return:.2f} | Global SUM={self.overall_sum_of_rewards:.2f}"
        )
        print(f"  - Predators: Total = {predator_total:.2f}")
        print(f"  - Prey:      Total = {prey_total:.2f}")

        for group, totals in group_rewards.items():
            print(f"  - {group}: Total = {sum(totals):.2f}")

        # Log episode length and LOS-rejected count using MetricsLogger
        metrics_logger.log_value("episode_length", episode_length, reduce="mean")
        los_rejected = self._episode_los_rejected.pop(episode_id, 0)
        metrics_logger.log_value("los_rejected_moves", los_rejected, reduce="mean")

        # --- Emit cooperation metrics to TensorBoard ---
        acc = self._coop_accumulators.pop(episode_id, None)
        if acc and acc.get("ai_n", 0) > 0:
            ai_raw = acc["ai_sum_raw"] / max(1, acc["ai_n"])
            metrics_logger.log_value("coop/ai_raw", ai_raw, reduce="mean")
            # Only log LOS-aware AI if env had it enabled (we only accumulated then)
            if acc.get("ai_sum_los", 0.0) > 0.0:
                ai_los = acc["ai_sum_los"] / max(1, acc["ai_n"])
                metrics_logger.log_value("coop/ai_los", ai_los, reduce="mean")

        kpa = self._kpa_state.pop(episode_id, None)
        if kpa:
            with_trials = max(0, int(kpa.get("with_trials", 0)))
            with_events = max(0, int(kpa.get("with_events", 0)))
            without_trials = max(0, int(kpa.get("without_trials", 0)))
            without_events = max(0, int(kpa.get("without_events", 0)))
            p_with = (with_events / with_trials) if with_trials else 0.0
            p_without = (without_events / without_trials) if without_trials else 0.0
            metrics_logger.log_value("coop/kpa_with", p_with, reduce="mean")
            metrics_logger.log_value("coop/kpa_without", p_without, reduce="mean")
            metrics_logger.log_value("coop/kpa", p_with - p_without, reduce="mean")
            # Useful counts for confidence in estimates
            metrics_logger.log_value("coop/with_trials", with_trials, reduce="mean")
            metrics_logger.log_value("coop/without_trials", without_trials, reduce="mean")

    def on_train_result(self, *, result, **kwargs):
        # Add training time metrics
        now = time.time()
        total_elapsed = now - self.start_time
        iter_num = result.get("training_iteration", 1)
        iter_time = now - self.last_iteration_time
        self.last_iteration_time = now

        result["timing/iter_minutes"] = iter_time / 60.0
        result["timing/avg_minutes_per_iter"] = total_elapsed / 60.0 / iter_num
        result["timing/total_hours_elapsed"] = total_elapsed / 3600.0
        # Optional: surface custom metric if available in learner results aggregation
        # (Ray will automatically aggregate logged metrics like los_rejected_moves across episodes)

    # ---- Compatibility helpers ----
    def _episode_last_infos(self, episode) -> dict:
        """
        Return a mapping of agent_id -> last info dict for this episode, with
        compatibility across RLlib API changes.
        """
        # Preferred: public getters
        for name in ("get_last_infos", "get_infos"):
            if hasattr(episode, name):
                try:
                    infos = getattr(episode, name)()
                    if isinstance(infos, dict):
                        return infos
                except Exception:
                    pass
        # Common attrs in some versions
        for name in ("last_infos", "infos"):
            if hasattr(episode, name):
                try:
                    infos = getattr(episode, name)
                    if isinstance(infos, dict):
                        return infos
                except Exception:
                    pass
        # Fallback: internal mapping used by older last_info_for implementations
        if hasattr(episode, "_agent_to_last_info"):
            try:
                mapping = getattr(episode, "_agent_to_last_info")
                if isinstance(mapping, dict):
                    return mapping
            except Exception:
                pass
        return {}
