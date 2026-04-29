import random

import pygame
import math
from TanksBattleEnv_v3.utils import *
from TanksBattleEnv_v3.models import *
import time
import numpy as np


def categorical_action(action_raw, action_dim):
    action_ = [0, 0, 0, 0]
    if 0 in action_dim:

        if action_raw[0] == 0:
            action_[0] = -5
        elif action_raw[0] == 1:
            action_[0] = 0
        else:
            action_[0] = 5
    else:
        action_[0] = 0

    if 1 in action_dim:
        if action_raw[1] == 0:
            action_[1] = -1
        elif action_raw[1] == 1:
            action_[1] = 0
        else:
            action_[1] = 1
    else:
        action_[1] = 0

    if 2 in action_dim:
        if action_raw[2] == 0:
            action_[2] = -1
        elif action_raw[2] == 1:
            action_[2] = 0
        else:
            action_[2] = 1
    else:
        action_[2] = 0

    if 3 in action_dim:
        if action_raw[3] == 0:
            action_[3] = 0
        else:
            action_[3] = 1
    else:
        action_[3] = 0

    return action_


def continuous_action(action_raw):
    action_ = copy(action_raw)
    action_[0] *= 5
    if action_raw[3] <= 0:
        action_[3] = 0
    else:
        action_[3] = 1
    return action_


class TanksBattleEnv:
    def __init__(self, headless=False, max_time=1000, speed_control=False, borderlines=False):
        # 初始化pygame
        pygame.init()
        self.headless = headless
        self.speed_control = speed_control  # True直接控制速度，False控制加速度
        # 设置屏幕尺寸
        self.screen_width, self.screen_height = 1000, 1000  # 单位米
        if not self.headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            self.screen = None

        self.tanks = []
        # 创建一些障碍物
        self.borderlines = borderlines
        # self.obstacles = [Obstacle(500, 700, 100, 100, headless),
        #                   Obstacle(400, 400, 100, 100, headless),
        #                   Obstacle(700, 500, 100, 100, headless)]
        if borderlines:
            self.obstacles = [Obstacle(0, 0, 1, 1000, headless, 'border'),
                              Obstacle(0, 0, 1000, 1, headless, 'border'),
                              Obstacle(0, 999, 1, 1000, headless, 'border'),
                              Obstacle(999, 0, 1000, 1, headless, 'border'),
                              Obstacle(400, 400, 200, 200, headless)]
        else:
            self.obstacles = [Obstacle(400, 400, 200, 200, headless)]
        self.lasers = []  # 存储当前屏幕上的所有激光
        self.max_time = max_time
        self.time_remain = self.max_time
        self.other_tank_list = None
        self.mission_time_remain = None
        self.missions = None
        # self.ally_list = None
        # self.enemy_list = None

    def init(self, missions, camps, positions, angles, tank_sizes):
        # tank_sizes = (30, 20)
        max_speed = 5
        # 创建坦克
        tank_num = 0
        for i, camp in enumerate(camps):
            self.tanks.append(Tank(tank_num, positions[i][0], positions[i][1], angles[i],
                                   tank_sizes[i][0], tank_sizes[i][1], camp, max_speed, self.headless))

        obs = [tank.get_obs(self.time_remain, missions[i]) for i, tank in enumerate(self.tanks)]
        trainable_dim = [tank.get_trainable_dim(missions[i]['type']) for i, tank in enumerate(self.tanks)]

        return obs, trainable_dim

    def reset(self, missions, camps, positions, angles, tank_sizes, obstacle_coords, textual_info=False, replace=True):
        max_speed = 5
        # tank_size = (30, 20)
        tank_num = 0
        self.tanks = []
        for i, camp in enumerate(camps):
            self.tanks.append(Tank(tank_num, positions[i][0], positions[i][1], angles[i],
                                   tank_sizes[i][0], tank_sizes[i][1], camp, max_speed, self.headless))

        self.mission_time_remain = [0 for _ in range(len(self.tanks))]
        self.missions = missions
        self.other_tank_list = []
        for tank in self.tanks:
            other_tank_list = [t for t in self.tanks]
            other_tank_list.remove(tank)
            self.other_tank_list.append(other_tank_list)
        if self.borderlines:
            self.obstacles = ([Obstacle(0, 0, 1, 1000, self.headless, 'border'),
                               Obstacle(0, 0, 1000, 1, self.headless, 'border'),
                               Obstacle(0, 999, 1000, 1, self.headless, 'border'),
                               Obstacle(999, 0, 1, 1000, self.headless, 'border')]
                              + [Obstacle(obstacle_coords[i][0], obstacle_coords[i][1], 50, 50, self.headless)
                                 for i in range(len(obstacle_coords))])
        else:
            self.obstacles = [Obstacle(obstacle_coords[i][0], obstacle_coords[i][1], 50, 50, self.headless)
                              for i in range(len(obstacle_coords))]

        self.lasers = []  # 存储当前屏幕上的所有激光
        self.time_remain = self.max_time
        if replace:
            for i, tank in enumerate(self.tanks):
                if missions[i]['type'] == 'hide':
                    while True:
                        while True:
                            tank.x = random.uniform(50, 950)
                            tank.y = random.uniform(50, 950)
                            if not any(rectangles_overlap(tank.get_rotated_points(), obj.get_rotated_points())
                                       for obj in self.obstacles + self.other_tank_list[i]):
                                break
                        tank.under_watch = False
                        for j, other_tank in enumerate(self.tanks):
                            if tank == other_tank:
                                continue
                            other_tank.update_radar(self.obstacles, self.other_tank_list[j])
                        if tank.under_watch:
                            # print(tank.x, tank.y)
                            break
                    tank.load_enemies(self.other_tank_list[i])

                elif missions[i]['type'] == 'aim':
                    while True:
                        while True:
                            tank.x = random.uniform(50, 950)
                            tank.y = random.uniform(50, 950)
                            if not any(rectangles_overlap(tank.get_rotated_points(), obj.get_rotated_points())
                                       for obj in self.obstacles + self.other_tank_list[i]):
                                break
                        tank.update_radar(self.obstacles, self.other_tank_list[i])
                        if 4 in tank.radar_data:
                            break

                elif missions[i]['type'] == 'attack':
                    # while True:
                    #     tank.x = random.uniform(50, 950)
                    #     tank.y = random.uniform(50, 950)
                    #     if not any(rectangles_overlap(tank.get_rotated_points(), obj.get_rotated_points())
                    #                for obj in self.obstacles + self.other_tank_list[i]):
                    #         break
                    # tank.under_watch = False
                    # for j, other_tank in enumerate(self.tanks):
                    #     if tank == other_tank:
                    #         continue
                    #     other_tank.update_radar(self.obstacles, self.other_tank_list[j])
                    # if tank.under_watch:
                    #     # print(tank.x, tank.y)
                    #     break
                    tank.load_enemies(self.other_tank_list[i])
        obs = [tank.get_obs(self.time_remain, missions[i]) for i, tank in enumerate(self.tanks)]
        if textual_info:
            tank_info = [tank.info for tank in self.tanks]
            return obs, tank_info
        else:
            return obs

    def assign_mission(self, missions):
        for i, mission in enumerate(missions):
            if mission is not None:
                self.missions[i] = mission
            self.mission_time_remain[i] = 1000
        obs = [tank.get_obs(self.time_remain, self.missions[i]) for i, tank in enumerate(self.tanks)]
        return obs

    def step(self, actions, missions, mission_check=None, textual_info=False, auto_fire=False):
        if textual_info:
            for tank in self.tanks:
                tank.reset_info()

        if mission_check is None:
            mission_check = [True for _ in range(len(self.tanks))]

        # 更新坦克机动、火炮状态
        for i, tank in enumerate(self.tanks):
            if tank.is_distroyed:
                continue
            laser = tank.act(actions[i], self.obstacles, self.other_tank_list[i])
            if laser:
                self.lasers.append(laser)

        # 更新坦克位置，检查是否与障碍物碰撞
        for i, tank in enumerate(self.tanks):
            if tank.is_distroyed:
                continue
            original_position = (tank.x, tank.y)
            tank.move(self.screen_width, self.screen_height, self.obstacles, self.other_tank_list[i])
            # 使用多边形碰撞检测
            collided = False
            for obj in self.other_tank_list[i] + self.obstacles:
                if polygon_collision(tank.get_rotated_points(), obj.get_rotated_points()):
                    collided = True
                    break

            if collided:
                tank.x, tank.y = original_position  # 如果发生碰撞，恢复原始位置

            tank.update_radar(self.obstacles, self.other_tank_list[i])
            # print(tank.radar_data)

        # 队内共享坦克信息
        for i, tank in enumerate(self.tanks):
            if tank.is_distroyed:
                continue
            tank.share_tank_info(self.other_tank_list[i], self.obstacles)

        # 更新和绘制激光，检查是否与障碍物或坦克碰撞
        for laser in self.lasers:
            laser.draw(self.screen)

        # 渲染
        if not self.headless:
            self.screen.fill((255, 255, 255))  # 清屏

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        for m in missions:
            if not self.headless and m['type'] == 'scout':
                pygame.draw.circle(self.screen, (173, 255, 47), m['target'], 40, 10)  # 绘制侦察目标点

        for i, tank in enumerate(self.tanks):
            tank.draw(self.screen, self.obstacles, self.other_tank_list[i])
            # if tank.aim_target_type == 4 and auto_fire:
            #     laser = tank.act([0, 0, 0, 1], self.obstacles, self.other_tank_list[i])
            #     if laser:
            #         self.lasers.append(laser)

        # 绘制激光
        for laser in self.lasers:
            laser.update()
            if laser.duration <= 0:
                self.lasers.remove(laser)
            else:
                laser.draw(self.screen)

        if not self.headless:
            pygame.event.pump()
            pygame.display.flip()

        self.time_remain -= 1
        self.mission_time_remain = [t - 1 for t in self.mission_time_remain]

        mission_terminal = [not check for check in mission_check]
        success = [False for _ in range(len(missions))]

        for i, tank in enumerate(self.tanks):
            mission_dict = missions[i]
            if tank.is_distroyed:
                mission_terminal[i] = True

            elif mission_dict['type'] == 'battle':
                camp = tank.camp
                if all([tank.is_distroyed for tank in self.tanks if tank.camp != camp]):
                    mission_terminal[i] = True
                    success[i] = True

            elif mission_dict['type'] == 'aim':
                if auto_fire:
                    if tank.aim_target_type == 5:
                        mission_terminal[i] = True
                        success[i] = True
                else:
                    if tank.aim_target_type == 4:
                        mission_terminal[i] = True
                        success[i] = True

            elif mission_dict['type'] == 'scout':
                if textual_info and tank.watch_enemy:
                    mission_terminal[i] = True
                    success[i] = True
                    tank.info += '发现敌人单位，侦察终止。'

                elif np.sqrt(
                        ((tank.x - mission_dict['target'][0]) ** 2 + (tank.y - mission_dict['target'][1]) ** 2)) <= 50:
                    mission_terminal[i] = True
                    success[i] = True
                    tank.info += '抵达目标位置，尚未发现敌人单位。'

            elif mission_dict['type'] == 'hide':
                hides = []
                obstacle_rects = [obj.get_rotated_points() for obj in self.obstacles if obj.type == 'house']
                # if self.under_watch:
                #     reward = -0.1
                for tank_dict in tank.enemy_last_seen:
                    if tank_dict['position'][0] != -1000:
                        hide = False
                        line = [[tank.x, tank.y], [tank_dict['position'][0], tank_dict['position'][1]]]
                        for rect in obstacle_rects:
                            if line_segment_rect_intersect(line, rect):
                                hide = True
                                break
                        hides.append(hide)
                if all(hides):
                    mission_terminal[i] = True
                    success[i] = True
                    tank.info += '已经规避所有敌人火力。'

            elif mission_dict['type'] == 'attack':
                if tank.aim_target_type == 4 and not textual_info:
                    mission_terminal[i] = True
                    success[i] = True

            if self.mission_time_remain[i] == 0:
                mission_terminal[i] = True
                mission_type = mission_dict['type']
                tank.info += f'限定时间内没有完成{mission_type}！'

            # if success[i]:
            #     mission_type = mission_dict['type']
            #     tank.info += f'{mission_type}任务成功！'

        terminal = all(mission_terminal)
        if self.time_remain == 0:
            terminal = True
        elif textual_info:
            terminal = False

        if self.speed_control:
            for tank in self.tanks:
                tank.speed = 0

        if not textual_info:
            observations = [tank.get_obs(self.time_remain, missions[i])
                            for i, tank in enumerate(self.tanks)]
            rewards = [tank.get_reward(self.other_tank_list[i], missions[i], self.obstacles)
                       for i, tank in enumerate(self.tanks)]
            return observations, rewards, terminal, success
        else:
            observations = [tank.get_obs(self.mission_time_remain[i], missions[i])
                            for i, tank in enumerate(self.tanks)]
            tank_info = [tank.info for tank in self.tanks]
            return observations, mission_terminal, terminal, tank_info

#
#
# if __name__ == '__main__':
#     mission = {'type': 'scout', 'target': (100, 100)}
#     env = TanksBattleEnv(headless=False, speed_control=True)
#     env.reset(True, 3)
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             # 用户控制坦克1
#         action = [0, 0, 0, 0]  # [加速度, 车体转向, 炮塔转向, 开火]
#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_w]:
#             action[0] = 1
#         if keys[pygame.K_s]:
#             action[0] = -1
#         if keys[pygame.K_a]:
#             action[1] = -1
#         if keys[pygame.K_d]:
#             action[1] = 1
#         if keys[pygame.K_q]:  # Q键用于控制坦克1的炮塔向左转
#             action[2] = -0.1
#         if keys[pygame.K_e]:  # E键用于控制坦克1的炮塔向右转
#             action[2] = 0.1
#         if keys[pygame.K_SPACE]:  # 空格键用于坦克1开火
#             action[3] = 1
#
#         s, r, t, w = env.step(action, mission)
#         # print(s)
#         print(r)
#         # print(t)
#
#         # print(env.tank2.is_distroyed)
#         # print(env.tank1.radar_data)
#         # time.sleep(0.1)
