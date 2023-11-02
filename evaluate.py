# Prevent numpy from using up all cpu
from math import fabs
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position
import argparse
from pathlib import Path
import numpy as np
import utils
import time
import frontier
import visualize
import environment
import munkres
import math
import random

def _run_eval(cfg, num_episodes=40):
    if 'frontier_exploration' not in cfg:
           cfg['frontier_exploration'] = False

    if 'random' not in cfg:
           cfg['random'] = False

    env = utils.get_env_from_cfg(cfg, random_seed=9)
    policy = utils.get_policy_from_cfg(cfg, env.get_action_space(), random_seed=9)
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    # Index [0] to ignore state_info
    states = [env.reset(robot_index)[0] for robot_index in range(env.num_agents)]
    start_time = time.time()
    end_time = time.time()
    done = False
    while True:
        if cfg['frontier_exploration']:
            # Get a list of points sampled from each frontier
            frontiers = frontier.find_frontier(env.global_visit_freq_map > 0, env.configuration_space + env.wall_map)
            # if not enough frontiers, add random
            for _ in range(env.num_agents - len(frontiers)):
                frontiers.append((random.randint(0, env.configuration_space.shape[0]-1), 
                                  random.randint(0, env.configuration_space.shape[1]-1)))
                
            # Get a list of robot pixel poses
            robot_poses = []
            for robot_index in range(env.num_agents):
                robot_position = env.robot_position[robot_index]
                pix = environment.position_to_pixel_indices(robot_position[0], robot_position[1], env.configuration_space.shape)
                robot_poses.append(pix)
            
            # Get a cost matrix of (Robot Pose X Frontier Point)
            matrix = np.empty((len(robot_poses), len(frontiers)))
            for i, rp in enumerate(robot_poses):
                for j, f in enumerate(frontiers):
                    matrix[i][j] = math.hypot(rp[0] - f[0], rp[1] - f[1])
            
            # Allocate each robot to a frontier while minimizing total cost
            indexes = munkres.Munkres().compute(matrix)

            # print('\t', frontiers)
            # print(robot_poses[0], matrix[0])
            # print(robot_poses[1], matrix[1])
            # print(indexes)
            indexes = sorted(indexes, key=lambda tup:tup[0])            

        infos = []
        for robot_index in range(env.num_agents):
            if cfg['frontier_exploration']:
                # For each robot, look up its assigned frontier point
                target = frontiers[indexes[robot_index][1]]
                # print(robot_index, target)
                curr_state, _, curr_done, info = env.step(target, robot_index, the_action_is_relative_pixels=True)
            
            elif cfg['random']:
                curr_state, _, curr_done, info = env.step(random.randint(0, env.get_action_space()-1), robot_index)
            else:
                action, _ = policy.step(states[robot_index])
                curr_state, _, curr_done, info = env.step(action, robot_index)
            
            infos.append(info)

            states[robot_index] = curr_state
            done = True if curr_done else done

        end_time = time.time()
        
        time_elasped = end_time - start_time

        exploration_ratio_goal = 0.5


        # Decide if and why the episode should end
        end_episode_condition = ''
        if 'theoretical_exploration' in cfg and cfg['theoretical_exploration']:
            # Complete upon reaching exploration_ratio_goal
            if info['ratio_explored'] > exploration_ratio_goal:
                end_episode_condition = 'theoretical_exploration'
        else:
            # Complete upon seeing cube, or done=True
            if done:
                end_episode_condition = 'found_block'
        if time_elasped > 600:
            end_episode_condition = 'timed_out'

        for info in infos:
            data[episode_count].append({'cube_found': info['cube_found'],\
                                        'cumulative_distance': info['cumulative_distance'],\
                                        'explored_area': info['explored_area'],\
                                        'overlapped_ratio': info['overlapped_ratio'],\
                                        'non_overlapped_ratio': info['non_overlapped_ratio'],\
                                        'ratio_explored': info['ratio_explored'],\
                                        'repetitive_exploration_rate': info['repetive_exploration_rate'],\
                                        'bandwidth': info['bandwidth'],
                                        'bandwidth_fast': info['bandwidth_fast'],
                                        'end_episode_condition': end_episode_condition})
            
        # Reset all agents and log statistics
        if end_episode_condition != '':
            for robot_index in range(env.num_agents):
                env.reset(robot_index)
            done = False

            start_time = time.time()
            episode_count += 1

            env._visualize_state_representation()

            if end_episode_condition == 'theoretical_exploration':
                print('Completed {}/{} episodes (theoretical exploration)'.format(episode_count, num_episodes))
            if end_episode_condition == 'found_block':
                print('Completed {}/{} episodes (found block)'.format(episode_count, num_episodes))
            if end_episode_condition == 'timed_out':
                print('Failed to complete episode:', episode_count)
                
            if episode_count >= num_episodes:
                break

    return data

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        print('Please provide a config path')
        return
    cfg = utils.read_config(config_path)

    cfg['use_gui'] = args.use_gui
    cfg['show_state_representation'] = args.show_state_representation
    cfg['show_occupancy_map'] = args.show_occupancy_map

    eval_dir = Path(cfg.logs_dir).parent / 'eval'
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    eval_path = eval_dir / '{} {}.npy'.format(cfg.run_name, time.time())
    data = _run_eval(cfg, num_episodes=200)
    print('saved eval to', eval_path)
    np.save(eval_path, data)
    print(eval_path)

    if 'num_agents' not in cfg:
        cfg['num_agents'] = 1

    visualize.visualize(eval_path, cfg['num_agents'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    parser.add_argument('--use_gui', action='store_true')
    parser.add_argument('--show_state_representation', action='store_true')
    parser.add_argument('--show_occupancy_map', action='store_true')
    main(parser.parse_args())
