import copy
import numpy as np
from alg_parameters import *
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import NoSolutionError

dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1),
           8: (-1, 1)}  # x,y operation for corresponding action
# -{0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST}
actionDict = {v: k for k, v in dirDict.items()}
opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

class State(object):  # world property
    def __init__(self, state,state_dict, goals,global_num_agents,start_list,goal_list,observation_size,eval=False):
        """initialization"""
        self.state = copy.deepcopy(state)  # dict position->tatic obstacle: -1,0: empty,positive integer (the number of agents)
        self.state_dict=copy.deepcopy(state_dict)  # dict position-> agent id
        self.goals = goals  # dict position-> empty: 0, goal = positive integer (agent_index)
        self.global_num_agents = global_num_agents
        self.agents_poss=copy.deepcopy(start_list)  # agent_index-1(agent index)->possition
        self.agents_goals = goal_list  #  agent_index-1(agent index)->possition
        if eval==False:
            self.get_heuri_map()
        self.observation_size=observation_size
        assert (len(self.agents_poss) == global_num_agents)

    def reset_local_tasks(self,state,state_dict,start_list,local_agents):
        self.state = copy.deepcopy(state)
        self.state_dict = copy.deepcopy(state_dict)
        self.agents_poss = copy.deepcopy(start_list)
        self.local_agents=local_agents  # index
        self.local_num_agents=len(local_agents)
        self.local_agents_poss=[]
        self.local_agents_goal=[]
        for i in local_agents:
            self.local_agents_poss.append(self.agents_poss[i])  # independent from agents poss
            self.local_agents_goal.append(self.agents_goals[i])

    def get_dir(self, action):
        """obtain corresponding x,y operation based on action"""
        return dirDict[action]

    def get_action(self, direction):
        """obtain corresponding action based on x,y operation"""
        return actionDict[direction]

    def local_task_done(self):
        """check if all agents on their goal"""
        num_complete = 0
        for i in self.local_agents:
            poss=self.agents_poss[i]
            if self.goals[poss[0], poss[1]] == i+1:
                num_complete += 1
        return num_complete == len(self.local_agents), num_complete

    def list_next_valid_actions(self, local_agent_index, prev_action=0):
        """obtain the valid actions that can not lead to colliding with obstacles and boundaries
        or backing to previous position at next time step"""
        available_actions = [0]  # staying still always allowed

        agent_pos = self.local_agents_poss[local_agent_index]
        ax, ay = agent_pos[0], agent_pos[1]

        for action in range(1, EnvParameters.N_ACTIONS):  # every action except 0
            direction = self.get_dir(action)
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.state.shape[0] or ax + dx < 0 or ay + dy >= self.state.shape[
                    1] or ay + dy < 0:  # out of boundaries
                continue
            if self.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                continue
            # otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:  # back to previous position
            available_actions.remove(opposite_actions[prev_action])
        return available_actions

    def get_heuri_map(self):
        dist_map = np.ones((self.global_num_agents, *self.state.shape), dtype=np.int32) * 2147483647
        for i in range(self.global_num_agents):  # iterate over all position for agents
            open_list = list()
            x, y = tuple(self.agents_goals[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up[0] >= 0 and self.state[up] != -1 and dist_map[
                    i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    if up not in open_list:
                        open_list.append(up)

                down = x + 1, y
                if down[0] < self.state.shape[0] and self.state[down] != -1 and dist_map[i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    if down not in open_list:
                        open_list.append(down)

                left = x, y - 1
                if left[1] >= 0 and self.state[left] != -1 and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    if left not in open_list:
                        open_list.append(left)

                right = x, y + 1
                if right[1] < self.state.shape[1] and self.state[right] != -1 and dist_map[i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    if right not in open_list:
                        open_list.append(right)

        self.heuri_map = np.zeros((self.global_num_agents, 4, *self.state.shape), dtype=np.bool)

        for x in range(self.state.shape[0]):
            for y in range(self.state.shape[1]):
                if self.state[x, y] != -1:  # empty
                    for i in range(self.global_num_agents):  # calculate relative distance

                        if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x - 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 0, x, y] = 1

                        if x < self.state.shape[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x + 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 1, x, y] = 1

                        if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y - 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 2, x, y] = 1

                        if y < self.state.shape[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y + 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 3, x, y] = 1

    def astar(self, world, start, goal, robots):
        """A* function for single agent"""
        for (i, j) in robots:
            world[i, j] = 1
        try:
            path = od_mstar.find_path(world, [start], [goal], inflation=1, time_limit=5)
        except NoSolutionError:
            path = None
        for (i, j) in robots:
            world[i, j] = 0
        return path

    def get_blocking_reward(self, local_agent_index):
        """calculates how many agents are prevented from reaching goal and returns the blocking penalty"""
        other_agents = []
        other_locations = []
        inflation = 10
        top_left = (self.local_agents_poss[local_agent_index][0] - self.observation_size // 2,
                    self.local_agents_poss[local_agent_index][1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        for agent in range(self.local_num_agents):
            if agent == local_agent_index:
                continue
            x, y = self.local_agents_poss[agent]
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agent not in FOV
                continue
            other_agents.append(agent)
            other_locations.append((x, y))

        num_blocking = 0
        world=(self.state == -1).astype(int)
        for agent in other_agents:   # local agent index
            other_locations.remove(self.local_agents_poss[agent])
            path_before = self.astar(world, self.local_agents_poss[agent], self.local_agents_goal[agent],
                                     robots=other_locations + [
                                         self.local_agents_poss[local_agent_index]])
            path_after = self.astar(world, self.local_agents_poss[agent], self.local_agents_goal[agent],
                                    robots=other_locations)
            other_locations.append(self.local_agents_poss[agent])
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                num_blocking += 1

        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking