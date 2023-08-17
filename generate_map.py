import copy
import random
import sys
import os

import numpy as np

def get_connected_region(world0, regions_dict, x0, y0):
    # ensure at the beginning of an episode, all agents and their goal at the same connected region
    sys.setrecursionlimit(1000000)
    if (x0, y0) in regions_dict:  # have done
        return regions_dict[(x0, y0)]
    visited = set()
    sx, sy = world0.shape[0], world0.shape[1]
    work_list = [(x0, y0)]
    while len(work_list) > 0:
        (i, j) = work_list.pop()
        if i < 0 or i >= sx or j < 0 or j >= sy:
            continue
        if world0[i, j] == -1:
            continue  # crashes
        if world0[i, j] > 0:
            regions_dict[(i,
                          j)] = visited  # regions_dict[(i, j)]  change with this visited, avoid repeat calculation for the agents at the same connection area
        if (i, j) in visited:
            continue
        visited.add((i, j))
        work_list.append((i + 1, j))
        work_list.append((i, j + 1))
        work_list.append((i - 1, j))
        work_list.append((i, j - 1))
    regions_dict[(x0, y0)] = visited
    return visited

if __name__ == '__main__':
    prob = 0.2
    size = 40
    global_num_agent = 500

    map = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
    fix_state = copy.deepcopy(map)
    fix_state_dict = {}
    for i in range(int(size)):
        for j in range(int(size)):
            fix_state_dict[i, j] = []

    # randomize the position of agents
    agent_counter = 1
    start_list = []
    while agent_counter <= global_num_agent:
        x, y = np.random.randint(0, size), np.random.randint(0, size)
        if fix_state[x, y] == 0:
            fix_state[x, y] += 1
            fix_state_dict[x, y].append(agent_counter)
            start_list.append((x, y))
            agent_counter += 1
    assert (sum(sum(fix_state)) == global_num_agent + sum(sum(map)))

    # randomize the position of goals
    goals = np.zeros((int(size), int(size))).astype(int)
    goal_counter = 1
    agent_regions = dict()
    goal_list = []
    while goal_counter <= global_num_agent:
        agent_pos = start_list[goal_counter - 1]
        valid_tiles = get_connected_region(fix_state, agent_regions, agent_pos[0], agent_pos[1])
        x, y = random.choice(list(valid_tiles))
        if goals[x, y] == 0 and fix_state[x, y] != -1:
            # ensure new goal does not at the same grid of old goals or obstacles
            goals[x, y] = goal_counter
            goal_list.append((x, y))
            goal_counter += 1

    if not os.path.exists('./maps'):
        os.makedirs('./maps')

    with open('./maps/eval_map.npy', 'wb') as f:
        np.save(f, map)
        np.save(f, fix_state)
        np.save(f, fix_state_dict)
        np.save(f, start_list)
        np.save(f, goal_list)
        np.save(f, goals)

    # from my_lns2 import run_pp
    # if not os.path.exists("./record_files"):
    #     os.makedirs("./record_files")
    # can_not_use, makespan,global_num_collison, paths = run_pp(map, start_list, goal_list,0)
    # print('test')

    #
    # with open('./maps/eval_map.npy', 'rb') as f:
    #     a = np.load(f)
    #     b = np.load(f)
    #     c = np.load(f,allow_pickle=True).item()
    #     d = np.load(f)
    #     e = np.load(f)
    #     f_g = np.load(f)
    # d=list(d)
    # for i in range(len(d)):
    #     d[i] = tuple(d[i])
    # print('test')

    #
    # np.save('./maps/map.npy', map)
    # np.save('./maps/fix_state.npy', fix_state)
    # np.save('./maps/fix_state_dict.npy', fix_state_dict)
    # np.save('./maps/start_list.npy', start_list)
    # np.save('./maps/goal_list.npy', goal_list)
    # np.save('./maps/goals.npy', goals)
    #
    # np.save('./maps/goals.npy', goals)
