import copy
import random
import sys

import gym
import numpy as np
#from gym.envs.classic_control import rendering
from matplotlib.colors import hsv_to_rgb

from my_lns2 import run_pp, adaptive_destroy
from world_property import State
from alg_parameters import *


class MAPFEnv(gym.Env):
    """map MAPF problems to a standard RL environment"""

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self,env_id,global_num_agents_range=EnvParameters.GLOBAL_N_AGENT, fov_size=EnvParameters.FOV_SIZE, size=EnvParameters.WORLD_SIZE,
                 prob=EnvParameters.OBSTACLE_PROB):
        """initialization"""
        self.global_num_agents_range = global_num_agents_range
        self.observation_size =fov_size
        self.SIZE = size  # size of a side of the square grid
        self.PROB = prob  # obstacle density
        self.env_id=env_id

        self.viewer = None

    def global_set_world(self):
        """randomly generate a new task"""

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
                    regions_dict[(i, j)] = visited  # regions_dict[(i, j)]  change with this visited, avoid repeat calculation for the agents at the same connection area
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x0, y0)] = visited
            return visited

        prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
                                    self.PROB[1])  # sample a value from triangular distribution
        size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
                                p=[.5, .25, .25])  # sample a value according to the given probability
        self.global_num_agent=int(np.random.choice([self.global_num_agents_range[0], self.global_num_agents_range[0] * .7 + self.global_num_agents_range[1] * .3,
                                                self.global_num_agents_range[0] * .3 + self.global_num_agents_range[1] * .7, self.global_num_agents_range[1]]))
        # prob = self.PROB
        # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
        self.map = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id
        self.fix_state=copy.deepcopy(self.map)
        self.fix_state_dict = {}
        for i in range(int(size)):
            for j in range(int(size)):
                self.fix_state_dict[i,j]=[]

        # randomize the position of agents
        agent_counter = 1
        self.start_list = []
        while agent_counter <= self.global_num_agent:
            x, y = np.random.randint(0, size), np.random.randint(0, size)
            if self.fix_state[x, y] == 0:
                self.fix_state[x, y] +=1
                self.fix_state_dict[x,y].append(agent_counter)
                self.start_list.append((x, y))
                agent_counter += 1
        assert(sum(sum(self.fix_state)) == self.global_num_agent + sum(sum(self.map)))

        # randomize the position of goals
        goals = np.zeros((int(size),int(size))).astype(int)
        goal_counter = 1
        agent_regions = dict()
        self.goal_list = []
        while goal_counter <= self.global_num_agent:
            agent_pos = self.start_list[goal_counter - 1]
            valid_tiles = get_connected_region(self.fix_state, agent_regions, agent_pos[0], agent_pos[1])
            x, y = random.choice(list(valid_tiles))
            if goals[x, y] == 0 and self.fix_state[x, y] != -1:
                # ensure new goal does not at the same grid of old goals or obstacles
                goals[x, y] = goal_counter
                self.goal_list.append((x,y))
                goal_counter += 1

        self.world = State(self.fix_state,self.fix_state_dict, goals, self.global_num_agent,self.start_list,self.goal_list,self.observation_size)

    def joint_move(self, actions):
        """simultaneously move agents and checks for collisions on the joint action """
        for i in range(self.global_num_agent): # move dynamic obstacles(never collide with static obstacles)
            if i not in self.world.local_agents:
                max_len=len(self.paths[i])
                if max_len<=self.time_step:  # self.time_step-1 is the last step always stay on it
                    continue
                else:
                    self.world.state[self.paths[i][self.time_step-1]]-=1
                    self.world.state_dict[self.paths[i][self.time_step-1]].remove(i+1)
                    self.world.state[self.paths[i][self.time_step]]+=1
                    self.world.state_dict[self.paths[i][self.time_step]].append(i+1)
                    self.world.agents_poss[i]=self.paths[i][self.time_step]

        past_position = copy.deepcopy(self.world.agents_poss)  # the current position of agents after moving
        agent_status = np.zeros(self.local_num_agents)  # use to determine rewards and invalid actions

        for local_i, i in enumerate(self.world.local_agents):
            direction = self.world.get_dir(actions[local_i])
            ax = self.world.local_agents_poss[local_i][0]
            ay = self.world.local_agents_poss[local_i][1]  # current position

            # Not moving is always allowed
            if direction == (0, 0):
                continue

            # Otherwise, let's look at the validity of the move
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.world.state.shape[0] or ax + dx < 0 or ay + dy >= self.world.state.shape[1] or ay + dy < 0:
                agent_status[i] = -1
                raise ValueError("out of boundaries")

            if self.world.state[ax + dx, ay + dy] < 0:
                agent_status[i] = -2
                raise ValueError("collide with static obstacles")

            self.world.state[ax, ay] -= 1  # clean previous position
            self.world.state_dict[ax, ay].remove(i + 1)

            self.world.agents_poss[i] = (ax + dx, ay + dy)  # update agent's current position
            self.world.local_agents_poss[local_i] = (ax + dx, ay + dy)
            self.world.state[ax + dx, ay + dy] += 1
            self.world.state_dict[ax + dx, ay + dy].append(i + 1)

        # -6(vertex collision with agents)>-5(swap collision with agents)>-4(vertex collision with dynamic obstacles)=-3(swap collision with dynamic obstacles)>1(reach goal)>
        # 2(leave goal)=-1(out of boundaries)=-2(collide with static obstacles)=0(nothing)
        for local_i, i in enumerate(self.world.local_agents):
            if self.world.state[self.world.local_agents_poss[local_i]] > 1:  # already has agents, vertix collison with dynamic obstacles
                collide_agents_id = self.world.state_dict[self.world.local_agents_poss[local_i]]
                for j in collide_agents_id:
                    if j-1!=i:
                        if j - 1 in self.world.local_agents:
                            agent_status[local_i] = -6  # vetrix collide with agents
                            break  # highest priority
                        else:
                            agent_status[local_i] = -4  # vetrix collide with dynamic obstacles

                if agent_status[local_i] == -6:
                    continue

            collide_agent_id = self.world.state_dict[past_position[i]]  # now=past
            if len(collide_agent_id)> 0:
                for j in collide_agent_id:
                    if j-1!=i:
                        if j - 1 in self.world.local_agents:  # past=now
                            past_poss =past_position[j-1]
                            if past_poss==self.world.local_agents_poss[local_i] and self.world.agents_poss[j-1]!=past_poss:
                                agent_status[local_i] = -5
                                break
                        else:
                            max_len=len(self.paths[j-1])
                            if max_len<=self.time_step:
                                continue
                            else:
                                past_poss=self.paths[j-1][self.time_step-1]
                                if past_poss== self.world.local_agents_poss[local_i] and past_poss!=self.paths[j-1][self.time_step]:
                                    agent_status[local_i] = -3

            if agent_status[local_i] !=0:
                continue

            if self.world.goals[self.world.local_agents_poss[local_i]] == i + 1:
                agent_status[local_i] = 1  # reach goal
                continue

            if self.world.goals[self.world.local_agents_poss[local_i]] != i + 1 and self.world.goals[past_position[i]] == i + 1:
                agent_status[local_i] = 2

        return agent_status

    def observe(self, local_agent_index):
        """return one agent's observation"""
        agent_index=self.world.local_agents[local_agent_index]
        top_left = (self.world.agents_poss[agent_index][0] - self.observation_size // 2,
                    self.world.agents_poss[agent_index][1] - self.observation_size // 2)  # (top, left)
        obs_shape = (self.observation_size, self.observation_size)
        goal_map = np.zeros(obs_shape)  # own goal
        local_poss_map = np.zeros(obs_shape)  # agents
        dynamic_poss_map = np.zeros(obs_shape)  # agents
        local_goals_map = np.zeros(obs_shape)  # other observable agents' goal
        dynamic_goals_map= np.zeros(obs_shape)
        obs_map = np.zeros(obs_shape)  # obstacle
        guide_map=np.zeros((4,obs_shape[0],obs_shape[1]))
        visible_agents = []

        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):  # left and right
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of boundaries
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    continue
                guide_map[:,i - top_left[0], j - top_left[1]] = self.world.heuri_map[agent_index, :, i, j]
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if (i,j)==self.world.agents_poss[agent_index]:
                    # own position
                    local_poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.goals[i, j] == agent_index+1:
                    # own goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0:
                    for item in self.world.state_dict[i,j]:
                    # other agents' positions
                        if item !=agent_index+1:
                            visible_agents.append(item)
                            if item-1 in self.world.local_agents:
                                local_poss_map[i - top_left[0], j - top_left[1]] = 1
                            else:
                                dynamic_poss_map[i - top_left[0], j - top_left[1]] = 1

        for agent_id in visible_agents:
            x, y = self.world.agents_goals[agent_id-1]
            # project the goal out of FOV to the boundary of FOV
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            if agent_id-1 in self.world.local_agents:
                local_goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1
            else:
                dynamic_goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.world.agents_goals[agent_index][0] - self.world.agents_poss[agent_index][0]  # distance on x axes
        dy = self.world.agents_goals[agent_index][1] - self.world.agents_poss[agent_index][1]  # distance on y axes
        mag = (dx ** 2 + dy ** 2) ** .5  # total distance
        if mag != 0:  # normalized
            dx = dx / mag
            dy = dy / mag

        return [local_poss_map,dynamic_poss_map, goal_map, local_goals_map, dynamic_goals_map,
                obs_map,guide_map[0],guide_map[1],guide_map[2],guide_map[3]], [dx, dy, mag]

    def joint_step(self, actions):
        """execute joint action and obtain reward"""
        self.time_step+=1
        action_status = self.joint_move(actions)
        for i in range(self.local_num_agents):
            self.local_path[i].append(self.world.local_agents_poss[i])

        blockings = np.zeros((1, self.local_num_agents), dtype=np.float32)
        rewards = np.zeros((1, self.local_num_agents), dtype=np.float32)
        obs = np.zeros((1, self.local_num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.local_num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        next_valid_actions = []
        num_blockings = 0
        leave_goals = 0
        num_dynamic_collide = 0
        num_agent_collide = 0
        #-6,-5,-4,-3,-2,-1,0,1,2
        for i in range(self.local_num_agents):
            if actions[i] == 0:  # staying still -6,-4,0,1
                if action_status[i] == 1:  # stayed on goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                    x, num_blocking = self.world.get_blocking_reward(i)
                    num_blockings += num_blocking
                    rewards[:, i] += x
                    if x < 0:
                        blockings[:, i] = 1
                elif action_status[i] == -6:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_agent_collide += 1
                elif action_status[i] == -4:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_dynamic_collide += 1
                else:
                    rewards[:, i] = EnvParameters.IDLE_COST

            else:  # moving -6,-5,-4,-3,0,2,1
                if action_status[i] == 1:  # reached goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                elif action_status[i] == -6 or action_status[i] == -5:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_agent_collide += 1
                elif action_status[i] == -4 or action_status[i] == -3:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_dynamic_collide += 1
                elif action_status[i] == 2:  # leave own goal
                    rewards[:, i] = EnvParameters.ACTION_COST
                    leave_goals += 1
                else:  # nothing happen
                    rewards[:, i] = EnvParameters.ACTION_COST

            state = self.observe(i)
            obs[:, i, :, :, :] = state[0]
            vector[:, i, : 3] = state[1]

            next_valid_actions.append(self.world.list_next_valid_actions(i, actions[i]))

        vector[:, :, -1] = actions
        done, num_on_goal = self.world.local_task_done()
        if num_on_goal > self.max_on_goal:
            self.max_on_goal = num_on_goal
        if self.time_step >= EnvParameters.EPISODE_LEN:
            done = True
        return obs, vector, rewards, done, next_valid_actions, blockings, num_blockings, \
            leave_goals, num_on_goal, self.max_on_goal, num_dynamic_collide,num_agent_collide

    def _global_reset(self):
        """restart a new task"""
        self.global_set_world()  # back to the initial situation
        self.selected_neighbor=0
        can_not_use,makespan,self.global_num_collison, self.paths=run_pp(self.map,self.start_list,self.goal_list,self.env_id)
        if makespan>EnvParameters.EPISODE_LEN:
            can_not_use=True
        return can_not_use,self.global_num_collison

    def _local_reset(self, local_num_agents,local_succ,first_time,ALNS):
        """restart a new task"""
        self.local_num_agents = local_num_agents
        self.max_on_goal = 0
        self.time_step=0
        temp_path = copy.deepcopy(self.paths)
        if local_succ:
            update_weight =True
            for local_i,i in enumerate(self.world.local_agents):
                temp_path[i]=self.local_path[local_i]
        else:
            update_weight = False
            if first_time==False:
                self.destroy_weights[self.selected_neighbor]=(1 - TrainingParameters.Destroy_factor)* self.destroy_weights[self.selected_neighbor]
            else:
                self.destroy_weights=np.ones(3)
        temp_global_num_collison=self.global_num_collison
        self.global_num_collison,self.destroy_weights,self.local_agents,global_succ,self.selected_neighbor,makespan=\
            adaptive_destroy(temp_path,self.local_num_agents,ALNS,self.global_num_collison,
                         self.destroy_weights,update_weight,self.selected_neighbor,self.env_id)
        if global_succ:
            return True,self.destroy_weights,self.global_num_collison,makespan
        if self.global_num_collison<=temp_global_num_collison and first_time==False and local_succ==True:
            for local_i, i in enumerate(self.local_agents):
                self.paths[i]=self.local_path[local_i]

        self.world.reset_local_tasks(self.fix_state,self.fix_state_dict,self.start_list,self.local_agents)

        self.local_path=[]
        for i in range(self.local_num_agents):
            self.local_path.append([self.world.local_agents_poss[i]])
        return False,self.destroy_weights,self.global_num_collison,makespan

    def list_next_valid_actions(self,local_agent_index):
     return self.world.list_next_valid_actions(local_agent_index)

# def create_rectangle(self, x, y, width, height, fill, permanent=False):
    #     """draw a rectangle to represent an agent"""
    #     ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
    #     rect = rendering.FilledPolygon(ps)
    #     rect.set_color(fill[0], fill[1], fill[2])
    #     rect.add_attr(rendering.Transform())
    #     if permanent:
    #         self.viewer.add_geom(rect)
    #     else:
    #         self.viewer.add_onetime(rect)
    #
    # def create_circle(self, x, y, diameter, size, fill, resolution=20):
    #     """draw a circle to represent a goal"""
    #     c = (x + size / 2, y + size / 2)
    #     dr = math.pi * 2 / resolution
    #     ps = []
    #     for i in range(resolution):
    #         x = c[0] + math.cos(i * dr) * diameter / 2
    #         y = c[1] + math.sin(i * dr) * diameter / 2
    #         ps.append((x, y))
    #     circ = rendering.FilledPolygon(ps)
    #     circ.set_color(fill[0], fill[1], fill[2])
    #     circ.add_attr(rendering.Transform())
    #     self.viewer.add_onetime(circ)

    def init_colors(self):
        """the colors of agents and goals"""
        c = {a + 1: hsv_to_rgb(np.array([a / float(self.global_num_agent), 1, 1])) for a in range(self.global_num_agent)}
        return c

    # def _render(self, mode='human', close=False, screen_width=800, screen_height=800, action_probs=None):
    #     if close:
    #         return
    #     # values is an optional parameter which provides a visualization for the value of each agent per step
    #     size = screen_width / max(self.world.state.shape[0], self.world.state.shape[1])
    #     colors = self.init_colors()
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         self.reset_renderer = True
    #     if self.reset_renderer:
    #         self.create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6), permanent=True)
    #         for i in range(self.world.state.shape[0]):
    #             start = 0
    #             end = 1
    #             scanning = False
    #             write = False
    #             for j in range(self.world.state.shape[1]):
    #                 if self.world.state[i, j] != -1 and not scanning:  # free
    #                     start = j
    #                     scanning = True
    #                 if (j == self.world.state.shape[1] - 1 or self.world.state[i, j] == -1) and scanning:
    #                     end = j + 1 if j == self.world.state.shape[1] - 1 else j
    #                     scanning = False
    #                     write = True
    #                 if write:
    #                     x = i * size
    #                     y = start * size
    #                     self.create_rectangle(x, y, size, size * (end - start), (1, 1, 1), permanent=True)
    #                     write = False
    #     for agent in range(1, self.num_agents + 1):
    #         i, j = self.world.get_pos(agent)
    #         x = i * size
    #         y = j * size
    #         color = colors[self.world.state[i, j]]
    #         self.create_rectangle(x, y, size, size, color)
    #         i, j = self.world.get_goal(agent)
    #         x = i * size
    #         y = j * size
    #         color = colors[self.world.goals[i, j]]
    #         self.create_circle(x, y, size, size, color)
    #         if self.world.get_goal(agent) == self.world.get_pos(agent):
    #             color = (0, 0, 0)
    #             self.create_circle(x, y, size, size, color)
    #     if action_probs is not None:
    #         for agent in range(1, self.num_agents + 1):
    #             # take the a_dist from the given data and draw it on the frame
    #             a_dist = action_probs[agent - 1]
    #             if a_dist is not None:
    #                 for m in range(EnvParameters.N_ACTIONS):
    #                     dx, dy = self.world.get_dir(m)
    #                     x = (self.world.get_pos(agent)[0] + dx) * size
    #                     y = (self.world.get_pos(agent)[1] + dy) * size
    #                     s = a_dist[m] * size
    #                     self.create_circle(x, y, s, size, (0, 0, 0))
    #     self.reset_renderer = False
    #     result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
    #     return result

if __name__ == '__main__':
    from model import Model
    import torch
    import os
    env = MAPFEnv(1)
    if not os.path.exists("./record_files"):
        os.makedirs("./record_files")
    can_not_use,global_num_collison=env._global_reset()
    global_task_solved,destroy_weights,global_num_collison, makespan=env._local_reset(8, False, True,True)
    prev_action = np.zeros(8)
    valid_actions = []
    obs = np.zeros((1, 8, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                   dtype=np.float32)
    vector = np.zeros((1, 8, NetParameters.VECTOR_LEN), dtype=np.float32)
    hidden_state = (
        torch.zeros((8, NetParameters.NET_SIZE)).to(torch.device('cpu')),
        torch.zeros((8, NetParameters.NET_SIZE)).to(torch.device('cpu')))
    for i in range(8):
        valid_action = env.list_next_valid_actions(i)
        s = env.observe(i)
        obs[:, i, :, :, :] = s[0]
        vector[:, i, : 3] = s[1]
        vector[:, i, -1] = prev_action[i]
        valid_actions.append(valid_action)

    model= Model(0,torch.device('cpu'))

    actions, ps, values, pre_block, output_state, num_invalid = \
        model.step(obs, vector, valid_actions, hidden_state, 8)
    obs, vector, rewards, done, next_valid_actions, blockings, num_blockings,leave_goals, num_on_goal, max_on_goal, \
    num_dynamic_collide, modify_actions=    env.joint_step(actions, model, 0, output_state, ps,valid_actions)

    global_task_solved, destroy_weights, global_num_collison = env._local_reset(8, False, False, True)

    print("testing")






