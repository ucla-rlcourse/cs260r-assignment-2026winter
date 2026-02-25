"""
Local evaluation for multi-agent racing. Supports loading multiple agents
and racing them against each other.

Usage:
    # Test single agent vs opponents
    python eval_local.py --agent-dirs agents/example_agent

    # Race two agents against each other
    python eval_local.py --agent-dirs agents/agent_A agents/agent_B --mode versus

    # Visualize on a specific map
    python eval_local.py --agent-dirs agents/example_agent --render --map hairpin
"""

import argparse
import importlib.util
import os
import sys

import numpy as np
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from env import OPPONENT_POLICIES
from racing_maps import RACING_MAPS, set_racing_map


def load_policy(agent_dir):
    agent_py = os.path.join(agent_dir, "agent.py")
    if not os.path.exists(agent_py):
        raise FileNotFoundError(f"No agent.py found in {agent_dir}")

    spec = importlib.util.spec_from_file_location(
        f"agent_{os.path.basename(agent_dir)}", agent_py
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.abspath(agent_dir))
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module.Policy()


def _compute_bev_size(env):
    """Compute BEV film size from the map bounding box."""
    bbox = env.current_map.road_network.get_bounding_box()
    x_ext = bbox[1] - bbox[0]
    y_ext = bbox[3] - bbox[2]
    film_px = int(np.ceil(1.15 * max(x_ext, y_ext))) + 10
    return (film_px, film_px)


def _render_bev(env, bev_size, display_size=(600, 600)):
    """Render a BEV topdown frame and show it in a cv2 window."""
    import cv2
    try:
        frame = env.render(
            mode="topdown",
            film_size=bev_size,
            screen_size=bev_size,
            scaling=1,
            target_agent_heading_up=False,
            center_on_map=True,
            window=False,
        )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, display_size)
        cv2.imshow("BEV", bgr)
        cv2.waitKey(1)
    except Exception:
        pass


def evaluate_single(agent_dir, num_episodes=5, num_agents=4,
                     opponent_policy="aggressive", render=False, seed=0):
    """Evaluate one agent against built-in opponents."""
    policy = load_policy(agent_dir)
    opp_fn = OPPONENT_POLICIES[opponent_policy]

    env = MultiAgentRacingEnv({
        "num_agents": num_agents,
        "use_render": render,
        "crash_done": False,
        "crash_vehicle_done": False,
        "out_of_road_done": True,
        "allow_respawn": False,
        "horizon": 3000,
        "start_seed": seed,
        "map_config": {
            "lane_num": max(2, num_agents),
            "exit_length": 20,
            "bottle_lane_num": max(4, num_agents),
            "neck_lane_num": 1,
            "neck_length": 20,
        },
    })
    ego_id = "agent0"
    results = {
        "rewards": [],
        "route_completions": [],
        "speeds": [],
        "win_count": 0,
        "lose_count": 0,
        "arrive_count": 0,
    }
    bev_size = None

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        if render and bev_size is None:
            bev_size = _compute_bev_size(env)
        policy.reset()
        ep_speeds = []
        ep_episode_reward = 0.0
        ep_route_completion = 0.0
        ep_arrive = False
        steps = 0
        opponent_obs = {k: v for k, v in obs_dict.items() if k != ego_id}
        ego_done = False

        while not ego_done:
            actions = {}
            for aid in list(env.agents.keys()):
                if aid == ego_id:
                    actions[aid] = policy(obs_dict[ego_id])
                else:
                    opp_obs = opponent_obs.get(aid, np.zeros(env.observation_space[aid].shape))
                    actions[aid] = opp_fn(opp_obs, aid)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            if render:
                _render_bev(env, bev_size)
            opponent_obs = {k: v for k, v in obs_dict.items() if k != ego_id}
            steps += 1

            # Collect per-step metrics from info
            ego_info = infos.get(ego_id, {})
            ep_speeds.append(ego_info.get("speed_km_h", 0.0))
            ep_episode_reward = ego_info.get("episode_reward", ep_episode_reward)
            if ego_info.get("arrive_dest", False) and not ep_arrive:
                ep_arrive = True

            # Read route completion from live vehicle
            if ego_id in env.agents:
                try:
                    ep_route_completion = env.agents[ego_id].navigation.route_completion
                except Exception:
                    pass

            ego_done = terms.get(ego_id, terms.get("__all__", False)) or \
                       truncs.get(ego_id, truncs.get("__all__", False))

        avg_speed = float(np.mean(ep_speeds)) if ep_speeds else 0.0
        results["rewards"].append(ep_episode_reward)
        results["route_completions"].append(ep_route_completion)
        results["speeds"].append(avg_speed)
        if ep_arrive:
            results["arrive_count"] += 1
            results["win_count"] += 1
        else:
            results["lose_count"] += 1

        print(f"  Ep {ep}: reward={ep_episode_reward:.2f}, "
              f"route={ep_route_completion:.2%}, "
              f"speed={avg_speed:.1f} km/h, "
              f"arrive={ep_arrive}")

    env.close()
    if render:
        import cv2
        cv2.destroyAllWindows()

    avg_reward = float(np.mean(results["rewards"]))
    avg_route = float(np.mean(results["route_completions"]))
    avg_speed = float(np.mean(results["speeds"]))

    print(f"\n--- Results ({num_episodes} episodes) ---")
    print(f"  avg_reward:           {avg_reward:.2f}")
    print(f"  avg_route_completion: {avg_route:.2%}")
    print(f"  avg_speed:            {avg_speed:.1f} km/h")
    print(f"  arrive_count:         {results['arrive_count']}/{num_episodes}")
    print(f"  win_count:            {results['win_count']}")
    print(f"  lose_count:           {results['lose_count']}")

    return {
        "avg_reward": avg_reward,
        "avg_route_completion": avg_route,
        "avg_speed": avg_speed,
        "win_count": results["win_count"],
        "lose_count": results["lose_count"],
        "rank": 1,
        "details": {
            "rewards": [float(x) for x in results["rewards"]],
            "route_completions": [float(x) for x in results["route_completions"]],
            "speeds": [float(x) for x in results["speeds"]],
            "arrive_count": results["arrive_count"],
        },
    }


def evaluate_versus(agent_dirs, num_episodes=5, render=False, seed=0):
    """Race multiple agents against each other."""
    num_agents = len(agent_dirs)
    policies = {}
    agent_ids = [f"agent{i}" for i in range(num_agents)]

    for i, agent_dir in enumerate(agent_dirs):
        policies[agent_ids[i]] = load_policy(agent_dir)
        print(f"  {agent_ids[i]} <- {agent_dir}")

    env = MultiAgentRacingEnv({
        "num_agents": num_agents,
        "use_render": render,
        "crash_done": False,
        "crash_vehicle_done": False,
        "out_of_road_done": True,
        "allow_respawn": False,
        "horizon": 3000,
        "start_seed": seed,
        "map_config": {
            "lane_num": max(2, num_agents),
            "exit_length": 20,
            "bottle_lane_num": max(4, num_agents),
            "neck_lane_num": 1,
            "neck_length": 20,
        },
    })

    results = {
        aid: {
            "rewards": [],
            "route_completions": [],
            "speeds": [],
            "win_count": 0,
            "lose_count": 0,
            "arrive_count": 0,
        }
        for aid in agent_ids
    }
    bev_size = None

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        if render and bev_size is None:
            bev_size = _compute_bev_size(env)
        for p in policies.values():
            p.reset()

        ep_speeds = {aid: [] for aid in agent_ids}
        ep_episode_rewards = {aid: 0.0 for aid in agent_ids}
        ep_route_completions = {aid: 0.0 for aid in agent_ids}
        ep_arrive = {aid: False for aid in agent_ids}
        ep_arrive_step = {aid: None for aid in agent_ids}
        step = 0
        done_all = False

        while not done_all:
            actions = {}
            for aid in list(env.agents.keys()):
                if aid in policies and aid in obs_dict:
                    actions[aid] = policies[aid](obs_dict[aid])
                else:
                    actions[aid] = np.array([0.0, 0.0], dtype=np.float32)

            obs_dict, rewards, terms, truncs, infos = env.step(actions)
            if render:
                _render_bev(env, bev_size)
            step += 1

            # Collect per-step metrics from info
            for aid in agent_ids:
                if aid in infos:
                    info = infos[aid]
                    ep_speeds[aid].append(info.get("speed_km_h", 0.0))
                    ep_episode_rewards[aid] = info.get(
                        "episode_reward", ep_episode_rewards[aid]
                    )
                    if info.get("arrive_dest", False) and not ep_arrive[aid]:
                        ep_arrive[aid] = True
                        ep_arrive_step[aid] = step

            # Read route completion from live vehicles
            for aid in agent_ids:
                if aid in env.agents:
                    try:
                        ep_route_completions[aid] = env.agents[aid].navigation.route_completion
                    except Exception:
                        pass

            done_all = terms.get("__all__", False) or truncs.get("__all__", False)

        # Record episode results
        for aid in agent_ids:
            avg_speed = float(np.mean(ep_speeds[aid])) if ep_speeds[aid] else 0.0
            results[aid]["rewards"].append(ep_episode_rewards[aid])
            results[aid]["route_completions"].append(ep_route_completions[aid])
            results[aid]["speeds"].append(avg_speed)
            if ep_arrive[aid]:
                results[aid]["arrive_count"] += 1

        # Determine episode winner: first to arrive wins; ties = all win;
        # nobody arrives = all lose
        arrived_slots = [aid for aid in agent_ids if ep_arrive[aid]]
        if arrived_slots:
            min_step = min(ep_arrive_step[aid] for aid in arrived_slots)
            winners = [aid for aid in arrived_slots if ep_arrive_step[aid] == min_step]
            losers = [aid for aid in agent_ids if aid not in winners]
            for aid in winners:
                results[aid]["win_count"] += 1
            for aid in losers:
                results[aid]["lose_count"] += 1
        else:
            for aid in agent_ids:
                results[aid]["lose_count"] += 1

        status_parts = []
        for aid in agent_ids:
            s = f"{aid}: route={ep_route_completions[aid]:.2%}, speed={results[aid]['speeds'][-1]:.1f} km/h"
            if ep_arrive[aid]:
                s += f" (arrived@step {ep_arrive_step[aid]})"
            status_parts.append(s)
        winner_str = ", ".join(winners) if arrived_slots else "none"
        print(f"  Ep {ep}: winner={winner_str} | " + " | ".join(status_parts))

    env.close()
    if render:
        import cv2
        cv2.destroyAllWindows()

    # Rank by win_count (desc), then avg_reward (desc) as tiebreaker
    avg_rewards = {aid: float(np.mean(results[aid]["rewards"])) for aid in agent_ids}
    ranked = sorted(
        agent_ids,
        key=lambda s: (results[s]["win_count"], avg_rewards[s]),
        reverse=True,
    )

    print(f"\n--- Race Results ({num_episodes} episodes) ---")
    formatted = []
    for rank_idx, aid in enumerate(ranked):
        r = results[aid]
        dir_idx = agent_ids.index(aid)
        avg_reward = float(np.mean(r["rewards"]))
        avg_route = float(np.mean(r["route_completions"]))
        avg_speed = float(np.mean(r["speeds"]))

        print(f"  #{rank_idx+1} {aid} ({os.path.basename(agent_dirs[dir_idx])}): "
              f"wins={r['win_count']}, losses={r['lose_count']}, "
              f"avg_reward={avg_reward:.2f}, "
              f"avg_route={avg_route:.2%}, "
              f"avg_speed={avg_speed:.1f} km/h, "
              f"arrive={r['arrive_count']}/{num_episodes}")

        formatted.append({
            "agent_dir": agent_dirs[dir_idx],
            "agent_slot": aid,
            "avg_reward": avg_reward,
            "avg_route_completion": avg_route,
            "avg_speed": avg_speed,
            "win_count": r["win_count"],
            "lose_count": r["lose_count"],
            "rank": rank_idx + 1,
            "details": {
                "rewards": [float(x) for x in r["rewards"]],
                "route_completions": [float(x) for x in r["route_completions"]],
                "speeds": [float(x) for x in r["speeds"]],
                "arrive_count": r["arrive_count"],
            },
        })

    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "versus"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=2,
                        help="Total agents (only for single mode)")
    parser.add_argument("--opponent-policy", type=str, default="aggressive")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", type=str, default="circuit",
                        choices=list(RACING_MAPS.keys()),
                        help="Racing map variant")
    args = parser.parse_args()

    restore = set_racing_map(args.map)
    try:
        if args.mode == "versus":
            if len(args.agent_dirs) < 2:
                print("Versus mode requires at least 2 agent directories")
                return
            print(f"Racing {len(args.agent_dirs)} agents on '{args.map}':")
            evaluate_versus(args.agent_dirs, args.num_episodes, args.render, args.seed)
        else:
            agent_dir = args.agent_dirs[0]
            print(f"Evaluating {agent_dir} vs {args.opponent_policy} on '{args.map}':")
            evaluate_single(agent_dir, args.num_episodes, args.num_agents,
                             args.opponent_policy, args.render, args.seed)
    finally:
        restore()


if __name__ == "__main__":
    main()
