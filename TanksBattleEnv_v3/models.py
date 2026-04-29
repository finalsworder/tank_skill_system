import pygame
import math
from TanksBattleEnv_v3.utils import *
from copy import copy
import numpy as np


# 激光类
class Laser:
    def __init__(self, start_x, start_y, angle, camp, headless):
        self.start_x = start_x
        self.start_y = start_y
        self.angle = angle
        if camp == 'red':
            self.color = (255, 140, 0)  # 激光的颜色
        else:
            self.color = (30, 144, 255)
        self.end_x = start_x + math.cos(math.radians(angle)) * 100000  # 初始很长的激光
        self.end_y = start_y + math.sin(math.radians(angle)) * 100000
        self.active = True  # 激光是否有效
        self.duration = 5  # 激光显示的帧数

        self.headless = headless

    def draw(self, screen):
        if not self.headless:
            if self.active:
                pygame.draw.line(screen, self.color, (self.start_x, self.start_y), (self.end_x, self.end_y), 3)

    # def calculate_collision(self, obstacles, other_tanks):
    #     laser_line = ((self.start_x, self.start_y), (self.end_x, self.end_y))
    #
    #     # 初始化最近碰撞点为激光原始终点
    #     closest_intersection = (self.end_x, self.end_y)
    #     closest_distance = math.hypot(self.end_x - self.start_x, self.end_y - self.start_y)
    #
    #     # 检查激光是否与障碍物或坦克相交
    #     for obj in obstacles + other_tanks:
    #         intersection = calculate_polygon_line_intersection(obj.get_rotated_points(), *laser_line)
    #         if intersection:
    #             distance = math.hypot(intersection[0] - self.start_x, intersection[1] - self.start_y)
    #             if distance < closest_distance:
    #                 closest_distance = distance
    #                 closest_intersection = intersection
    #                 if obj in other_tanks:
    #                     obj.is_distroyed = True  # 如果激光击中坦克，将其标记为被摧毁

    # # 更新激光终点为最近的碰撞点
    # self.end_x, self.end_y = closest_intersection
    # if closest_distance < 100000:  # 假设的激光最大长度
    #     self.active = False

    def update(self):
        # 每帧减少持续时间
        self.duration -= 1


class virtual_Tank:
    def __init__(self, x, y, body_angle, width, height):
        self.x = x
        self.y = y
        self.body_angle = body_angle
        self.width = width
        self.height = height

    def get_rotated_points(self):
        """返回旋转后的矩形的四个角的坐标"""
        cx, cy = self.x + self.width / 2, self.y + self.height / 2
        points = [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]
        return [rotate_point(x, y, self.body_angle, cx, cy) for x, y in points]


# 坦克类
class Tank:
    def __init__(self, id, x, y, body_angle, width, height, camp, max_speed, headless, num_allies=2, num_enemies=3):

        self.id = id
        self.obs = None
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.camp = camp
        if camp == 'red':
            self.color = (255, 48, 48)
            self.turret_color = (255, 48, 48)
        elif camp == 'blue':
            self.color = (0, 191, 255)
            self.turret_color = (0, 191, 255)
        else:
            print('未定义的阵营，涂黑')
            self.color = (0, 0, 0)
            self.turret_color = (0, 0, 0)
        self.speed = 0
        self.body_angle = body_angle
        self.max_speed = max_speed
        self.turret_angle = 0  # 炮塔相对于车身的角度
        self.aim_angle = body_angle + 0  # 炮管的绝对角度
        self.is_distroyed = 0  # 增加坦克是否被摧毁的状态
        self.original_rect = pygame.Rect(x-width/2, y-height/2, width, height)  # 原始未旋转的矩形

        self.radar_lines = 72  # 雷达射线数量
        self.detection_range = 10000  # 雷达探测范围
        self.headless = headless
        # 默认最大队友，敌人数
        self.num_allies, self.num_enemies = num_allies, num_enemies

        self.radar_data = [0 for _ in range(self.radar_lines)]  # 雷达探测结果
        self.radar_distance = [0 for _ in range(self.radar_lines)]
        self.aim_target_type = 0
        self.enemy_last_seen = [
            {'position': [-1000, -1000], 'speed': 0, 'body_angle': 0, 'aim_angle': 0, 'distroyed': 0, 'seen': 0}
            for _ in range(num_enemies)]  # 'position': [-1000, -1000]在转换为观测时会除以1000
        self.ally_info = [{'position': [-1000, -1000], 'speed': 0, 'body_angle': 0, 'aim_angle': 0, 'distroyed': 0}
                          for _ in range(num_allies)]  # 'position': [-1000, -1000]在转换为观测时会除以1000
        self.watch_enemy = False
        self.see_enemy = False
        self.under_watch = False
        self.seen = False
        self.hit = False
        self.hit_obj = None
        self.fire_interval = 50
        self.fire_count_down = 0

        self.current_mission = None
        self.info = f'作战开始，当前位于坐标({int(self.x)}, {int(self.y)})，尚未发现敌人。'

    def reset_info(self):
        self.info = f'当前位于坐标({int(self.x)}, {int(self.y)})。'
        if not self.see_enemy:
            self.info = '尚未发现敌人。'

    def load_enemies(self, other_tanks):
        enemies = [tank for tank in other_tanks if tank.camp != self.camp]
        for i, tank in enumerate(enemies):
            if tank.camp != self.camp:
                self.enemy_last_seen[i]['position'] = [tank.x, tank.y]
                self.enemy_last_seen[i]['speed'] = tank.speed
                self.enemy_last_seen[i]['body_angle'] = tank.body_angle
                self.enemy_last_seen[i]['aim_angle'] = tank.aim_angle
                self.enemy_last_seen[i]['distroyed'] = tank.is_distroyed

    def get_trainable_dim(self, mission_type):
        if mission_type == 'battle':
            return [0, 1, 2, 3]  # 返回全部的动作维度
        elif mission_type == 'scout':
            return [0, 1]  # 返回机动相关的动作维度
        elif mission_type == 'aim':
            return [1, 2]
        elif mission_type == 'hide':
            return [0, 1]
        elif mission_type == 'attack':
            return [0, 1, 2]
        else:
            return [0, 1, 2, 3]  # 返回全部的动作维度

    def draw(self, screen, obstacles, other_tanks):
        # 绘制坦克身体
        points = self.get_rotated_points()
        if not self.headless:
            pygame.draw.polygon(screen, self.color, points)
            pygame.draw.polygon(screen, (0, 0, 0), points, 2)  # 绘制黑色边框

        # 如果坦克未被摧毁，绘制炮塔
        if not self.is_distroyed:
            self.draw_turret(screen)
            self.draw_aiming_line(screen, obstacles, other_tanks)

    def update_radar(self, obstacles, other_tanks):
        self.radar_data = [0] * self.radar_lines  # 重置雷达数据
        self.radar_distance = [-1000] * self.radar_lines  # 重置雷达距离数据
        self.watch_enemy = False
        for i in range(self.radar_lines):
            angle = 360 / self.radar_lines * i + self.body_angle
            end_x = self.x + math.cos(math.radians(angle)) * self.detection_range
            end_y = self.y + math.sin(math.radians(angle)) * self.detection_range
            radar_line = (self.x, self.y), (end_x, end_y)

            # 初始化为最大探测范围
            closest_intersection = None
            closest_distance = self.detection_range
            # min_distance = self.detection_range
            contact_type = 0  # 0表示没有接触

            # 检查雷达线是否与障碍物或坦克相交
            detected_obj = None
            for obj in obstacles + other_tanks:
                # if obj != self:  # 排除自身
                intersection = calculate_polygon_line_intersection(obj.get_rotated_points(), *radar_line)
                if intersection:
                    distance = math.hypot(intersection[0] - self.x, intersection[1] - self.y)
                    if distance < closest_distance:
                        closest_distance = distance
                        if obj in obstacles:
                            contact_type = 1
                        elif obj.camp == self.camp and obj.is_distroyed:
                            contact_type = 2  # 被击毁的队友坦克
                        elif obj.camp == self.camp and not obj.is_distroyed:
                            contact_type = 3  # 未被击毁的队友坦克
                        elif not obj.is_distroyed:
                            contact_type = 4  # 未被击毁的敌方坦克
                            detected_enemy = obj
                        else:
                            contact_type = 5  # 被击毁的敌方坦克

            self.radar_data[i] = contact_type
            if contact_type:
                self.radar_distance[i] = closest_distance

            if contact_type == 4:
                self.watch_enemy = True
                # self.enemy_last_seen[obj.id] = {'position': [obj.x, obj.y],
                #                                 'speed': obj.speed, 'body_angle': obj.body_angle,
                #                                 'aim_angle': obj.aim_angle}
                detected_enemy.under_watch = True
                detected_enemy.seen = True
        return

    def share_tank_info(self, other_tanks, obstacles=None):
        allies = [tank for tank in other_tanks if tank.camp == self.camp]
        enemies = [tank for tank in other_tanks if tank.camp != self.camp]

        for i, ally in enumerate(allies):
            self.ally_info[i] = {'position': [ally.x, ally.y],
                                 'speed': ally.speed, 'body_angle': ally.body_angle,
                                 'aim_angle': ally.aim_angle, 'distroyed': ally.is_distroyed}
        # self.see_enemy = False
        for i, enemy in enumerate(enemies):
            if enemy.under_watch:
                self.see_enemy = True
                self.enemy_last_seen[i] = {'position': [enemy.x, enemy.y],
                                           'speed': enemy.speed, 'body_angle': enemy.body_angle,
                                           'aim_angle': enemy.aim_angle, 'distroyed': enemy.is_distroyed, 'seen': 1,
                                           'obj': enemy}
                self.info += f'当前时刻在坐标({int(enemy.x)}, {int(enemy.y)})处发现敌方单位。'
                if obstacles is not None:
                    hiding = False
                    for item in obstacles:
                        if line_segment_rect_intersect([[self.x, self.y], [enemy.x, enemy.y]],
                                                       item.get_rotated_points()):
                            hiding = False
                            break
                    if not hiding:
                        enemy_to_self = aiming_deviation(self.x, self.y, self.aim_angle, enemy.x, enemy.y)
                        self_to_enemy = aiming_deviation(enemy.x, enemy.y, enemy.aim_angle, self.x, self.y)
                        if enemy_to_self < self_to_enemy:
                            self.info += '可以先手瞄准射击敌人。'
                        else:
                            self.info += '会被敌人先手瞄准射击。'
        # if not self.see_enemy:
        #     self.info = '尚未发现敌方单位。'

    def draw_turret(self, screen):
        # 计算炮塔中心位置
        turret_center_x = self.x
        turret_center_y = self.y

        # 绘制炮塔（圆形）
        if not self.headless:
            pygame.draw.circle(screen, self.turret_color, (int(turret_center_x), int(turret_center_y)), self.width // 4)
            pygame.draw.circle(screen, (0, 0, 0), (int(turret_center_x), int(turret_center_y)), self.width // 4,
                               2)  # 绘制黑色边框

        # 计算炮管末端位置
        turret_end_x = turret_center_x + math.cos(math.radians(self.aim_angle)) * self.width // 1.5
        turret_end_y = turret_center_y + math.sin(math.radians(self.aim_angle)) * self.width // 1.5

        # 绘制炮管
        if not self.headless:
            pygame.draw.line(screen, (0, 0, 0), (turret_center_x, turret_center_y), (turret_end_x, turret_end_y), 3)

    def update_turret_angle(self, angle_change):
        self.turret_angle += angle_change  # 更新相对角度
        self.aim_angle = (self.body_angle + self.turret_angle) % 360  # 更新绝对角度

    def get_rotated_points(self):
        """返回旋转后的矩形的四个角的坐标"""
        cx, cy = self.x, self.y
        points = [
            (self.x - self.width/2, self.y - self.height/2),
            (self.x + self.width/2, self.y - self.height/2),
            (self.x + self.width/2, self.y + self.height/2),
            (self.x - self.width/2, self.y + self.height/2)
        ]
        return [rotate_point(x, y, self.body_angle, cx, cy) for x, y in points]

    def check_collision_with_polygon(self, other_polygon):
        """检查该坦克与另一个多边形的碰撞"""
        for point in self.get_rotated_points():
            if point_inside_polygon(point, other_polygon):
                return True
        return False

    def draw_aiming_line(self, screen, obstacles, other_tanks):
        # 计算瞄准线的起点和终点
        # start_x, start_y = self.x, self.y
        start_x = self.x + math.cos(math.radians(self.aim_angle)) * self.width // 10
        start_y = self.y + math.sin(math.radians(self.aim_angle)) * self.width // 10
        end_x = start_x + math.cos(math.radians(self.aim_angle)) * 10000  # 假设1000是足够长的距离
        end_y = start_y + math.sin(math.radians(self.aim_angle)) * 10000

        # 初始化为最长的瞄准线
        closest_intersection = None
        closest_distance = float('inf')
        hit_obj = None
        self.aim_target_type = 0  # 重置目标类型

        # 检查瞄准线是否与障碍物或坦克相交
        for obj in obstacles + other_tanks:
            # if self != obj:  # 避免与自身碰撞
            if polygon_line_collision(obj.get_rotated_points(), (start_x, start_y), (end_x, end_y)):
                intersection = calculate_polygon_line_intersection(obj.get_rotated_points(),
                                                                   (start_x, start_y), (end_x, end_y))
                if intersection:
                    distance = math.hypot(intersection[0] - start_x, intersection[1] - start_y)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_intersection = intersection
                        hit_obj = obj

        # 更新瞄准线终点
        if closest_intersection:
            end_x, end_y = closest_intersection
            if hit_obj in obstacles:
                self.aim_target_type = 1
            elif hit_obj.camp == self.camp and hit_obj.is_distroyed:
                self.aim_target_type = 2  # 被击毁的队友坦克
            elif hit_obj.camp == self.camp and not hit_obj.is_distroyed:
                self.aim_target_type = 3  # 未被击毁的队友坦克
            elif not hit_obj.is_distroyed:
                self.aim_target_type = 4  # 未被击毁的敌方坦克
            else:
                self.aim_target_type = 5  # 被击毁的敌方坦克
        # 绘制瞄准线
        if not self.headless:
            pygame.draw.line(screen, (0, 255, 0), (start_x, start_y), (end_x, end_y), 1)  # 使用绿色绘制瞄准线

    def move(self, screen_width, screen_height, obstacles, tanks):
        # 如果坦克被摧毁，则不移动
        if self.is_distroyed:
            return

            # 计算新的位置
        new_x = self.x + self.speed * math.cos(math.radians(self.body_angle))
        new_y = self.y + self.speed * math.sin(math.radians(self.body_angle))

        # 暂时更新位置以检测碰撞
        original_position = (self.x, self.y)
        self.x, self.y = new_x, new_y
        collided = False

        # 检查是否与障碍物或其他坦克发生碰撞
        for obstacle in obstacles:
            if polygon_collision(self.get_rotated_points(), obstacle.get_rotated_points()):
                collided = True
                break

        for other_tank in tanks:
            if other_tank != self and polygon_collision(self.get_rotated_points(), other_tank.get_rotated_points()):
                collided = True
                break

        # 如果发生碰撞，恢复原始位置
        if collided or not (0 <= new_x <= screen_width - self.width and 0 <= new_y <= screen_height - self.height):
            self.x, self.y = original_position
        else:
            # 更新坐标
            self.x, self.y = new_x, new_y

    def fire(self):
        # 计算炮管末端位置作为激光的起始位置        end_x = start_x + math.cos(math.radians(self.aim_angle)) * 10000  # 假设1000是足够长的距离
        turret_center_x = self.x
        turret_center_y = self.y
        turret_end_x = turret_center_x + math.cos(math.radians(self.aim_angle)) * self.width // 10
        turret_end_y = turret_center_y + math.sin(math.radians(self.aim_angle)) * self.width // 10
        return Laser(turret_end_x, turret_end_y, self.aim_angle, self.camp, self.headless)

    def update_speed(self, acceleration):
        # 更新速度，但不超过最大速度
        self.speed += acceleration
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < -self.max_speed:
            self.speed = -self.max_speed

    def act(self, action, obstacles, other_tanks):
        if self.is_distroyed:
            return
        self.under_watch = False
        self.hit = False
        self.update_speed(action[0])
        self.body_angle += action[1]
        self.update_turret_angle(action[2])

        self.fire_count_down = max(0, self.fire_count_down - 1)
        if action[3] and self.fire_count_down == 0:
            laser = self.fire()
            self.fire_count_down = self.fire_interval
            # 发射时立即计算碰撞
            closest_intersection = None
            closest_distance = float('inf')
            hit_obj = None
            for obj in obstacles + other_tanks:
                # if self != obj:  # 避免与自身碰撞
                if polygon_line_collision(obj.get_rotated_points(), (laser.start_x, laser.start_y),
                                          (laser.end_x, laser.end_y)):
                    intersection = calculate_polygon_line_intersection(obj.get_rotated_points(),
                                                                       (laser.start_x, laser.start_y),
                                                                       (laser.end_x, laser.end_y))
                    if intersection:
                        distance = math.hypot(intersection[0] - laser.start_x, intersection[1] - laser.start_y)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_intersection = intersection
                            hit_obj = obj
            # 如果找到交点，更新激光终点
            if closest_intersection:
                laser.end_x, laser.end_y = closest_intersection
                laser.active = True

            # 检查是否击中坦克
            if isinstance(hit_obj, Tank):
                hit_obj.is_distroyed = True
                self.hit = True
                self.hit_obj = hit_obj
                if self.camp == hit_obj.camp:
                    self.info += f'击毁位于({hit_obj.x}, {hit_obj.y})处的友方坦克。'
                else:
                    self.info += f'击毁位于({hit_obj.x}, {hit_obj.y})处的敌方坦克。'

            return laser
        else:
            return None

    # def get_rect(self):
    #     return pygame.Rect(self.x, self.y, self.width, self.height)

    def get_obs(self, t, mission_dict, obstacles=None):
        if self.is_distroyed:
            return None
        ally_obs = []
        for i, ally in enumerate(self.ally_info):
            ally_obs += [ally['position'][0] / 1000, ally['position'][1] / 1000,
                         ally['speed'], ally['body_angle'] / 360,
                         ally['aim_angle'] / 360, ally['distroyed']]
            if ally['position'][0] >= 0 and not ally['distroyed']:
                x = ally['position'][0]
                y = ally['position'][1]
                body_angle = ally['body_angle']
                aim_angle = ally['aim_angle']
                self.info += f'队友{i}状态：坐标({int(x)}, {int(y)})'

        enemy_obs = []
        self.see_enemy = False
        for i, enemy in enumerate(self.enemy_last_seen):
            enemy_obs += [enemy['position'][0] / 1000, enemy['position'][1] / 1000,
                          enemy['speed'], enemy['body_angle'] / 360,
                          enemy['aim_angle'] / 360, enemy['distroyed']]
            if enemy['position'][0] >= 0 and not enemy['distroyed'] and enemy['seen']:
                x = enemy['position'][0]
                y = enemy['position'][1]
                body_angle = enemy['body_angle']
                aim_angle = enemy['aim_angle']
                self.info += f'在坐标({int(x)}, {int(y)})处探测到敌方单位{i}。'
                self.see_enemy = True

        if mission_dict['type'] == 'battle':
            self.obs = ([t / 1e3, self.x / 1e3, self.y / 1e3, self.speed / self.max_speed, self.body_angle / 360,
                         self.aim_angle / 360, self.fire_count_down / self.fire_interval, self.aim_target_type]
                        + self.radar_data + [d / 1e3 for d in self.radar_distance] + ally_obs + enemy_obs)
        elif mission_dict['type'] == 'aim':
            self.obs = ([t / 1e3, self.x / 1e3, self.y / 1e3, self.body_angle / 360,
                         self.aim_angle / 360, self.aim_target_type]
                        + self.radar_data + enemy_obs)
        elif mission_dict['type'] == 'scout':
            # self.obs = ([t / 1000, self.x / 1000, self.y / 1000, self.speed / self.max_speed, self.body_angle / 360,
            #              self.aim_angle / 360, self.fire_count_down / self.fire_interval, self.aim_target_type]
            #             + self.radar_data + [d / 1e3 for d in self.radar_distance] + ally_obs + enemy_obs
            #             + [(mission_dict['target'][0] - self.x) / 1000, (mission_dict['target'][1] - self.y) / 1000])
            dx = mission_dict['target'][0] - self.x
            dy = mission_dict['target'][1] - self.y
            angle_to_target = math.degrees(math.atan2(dy, dx)) % 360
            body_angle = self.body_angle % 360
            # print(dx, dy, math.degrees(math.atan(dy/dx)), body_angle, angle_to_target)
            angle_diff = min(abs(angle_to_target - body_angle), 360 - abs(angle_to_target - body_angle))
            self.obs = ([t / 1e3, self.x / 1e3, self.y / 1e3, self.body_angle / 360]
                        + [d / 1e3 for d in self.radar_distance]
                        + [(mission_dict['target'][0] - self.x) / 1000, (mission_dict['target'][1] - self.y) / 1000,
                           angle_diff / 180])
        elif mission_dict['type'] == 'hide':
            self.obs = ([t / 1e3, self.x / 1e3, self.y / 1e3, self.speed / self.max_speed, self.body_angle / 360,
                         self.aim_target_type] + self.radar_data + [d / 1e3 for d in self.radar_distance]
                        + ally_obs + enemy_obs)
        elif mission_dict['type'] == 'attack':
            self.obs = ([t / 1e3, self.x / 1e3, self.y / 1e3, self.speed / self.max_speed, self.body_angle / 360,
                         self.aim_angle / 360, self.fire_count_down / self.fire_interval, self.aim_target_type]
                        + self.radar_data + [d / 1e3 for d in self.radar_distance] + enemy_obs)

        return self.obs

    def get_reward(self, other_tanks, mission_dict, obstacles):
        # if self.is_distroyed:
        #     return None
        reward = 0

        if mission_dict['type'] == 'battle':
            enemies = [tank for tank in other_tanks if tank.camp != self.camp]
            reward = 0
            # 发现敌人奖励
            if self.watch_enemy:
                reward += 0.1

                # 瞄准敌人奖励

                if self.aim_target_type == 4:
                    reward += 0.1

                min_angle_diff = 180
                for enemy in enemies:
                    angle_to_enemy = math.atan2(enemy.y - self.y, enemy.x - self.x)
                    angle_to_enemy = math.degrees(angle_to_enemy) % 360
                    turret_angle = self.aim_angle % 360

                    # 计算夹角
                    angle_diff = min(abs(turret_angle - angle_to_enemy), 360 - abs(turret_angle - angle_to_enemy))
                    if angle_diff < min_angle_diff:
                        min_angle_diff = angle_diff
                # 根据夹角的大小给予奖励（夹角越小，奖励越大）
                reward += (1 - (min_angle_diff / 180) ** 1) * 0.1  # 例如，夹角小于30度时给予奖励

            if self.hit:
                if self.hit_obj.camp == self.camp:
                    reward -= 500  # 击毁友方奖励
                else:
                    reward += 500  # 击毁敌方奖励

            if self.is_distroyed:
                reward = -500

        elif mission_dict['type'] == 'aim':
            reward = 0
            enemies = [tank for tank in other_tanks if tank.camp != self.camp]
            # print(self.radar_data)

            # if 4 in self.radar_data:
            if self.aim_target_type == 4:
                reward += 1

            min_angle_diff = 180
            for enemy in enemies:
                angle_to_enemy = math.atan2(enemy.y - self.y, enemy.x - self.x)
                angle_to_enemy = math.degrees(angle_to_enemy) % 360
                turret_angle = self.aim_angle % 360

                # 计算夹角
                angle_diff = min(abs(turret_angle - angle_to_enemy), 360 - abs(turret_angle - angle_to_enemy))
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
            # 根据夹角的大小给予奖励（夹角越小，奖励越大）
            reward -= (min_angle_diff / 180) ** 1 * 0.1  # 例如，夹角小于30度时给予奖励

        elif mission_dict['type'] == 'scout':
            dx = mission_dict['target'][0] - self.x
            dy = mission_dict['target'][1] - self.y
            angle_to_target = math.degrees(math.atan2(dy, dx)) % 360
            body_angle = self.body_angle % 360
            # print(dx, dy, math.degrees(math.atan(dy/dx)), body_angle, angle_to_target)
            angle_diff = min(abs(angle_to_target - body_angle), 360 - abs(angle_to_target - body_angle))
            reward = - np.sqrt(dx ** 2 + dy ** 2) / 1000 - 0 * angle_diff / 180

            min_dist = min([dist for dist in self.radar_distance if dist > 0] + [1000])
            if min_dist < 8:
                reward -= 1

            if np.sqrt(((self.x - mission_dict['target'][0]) ** 2 + (self.y - mission_dict['target'][1]) ** 2)) <= 30:
                reward = 1

        elif mission_dict['type'] == 'hide':
            hides = []
            obstacle_rects = [obj.get_rotated_points() for obj in obstacles if obj.type == 'house']
            # if self.under_watch:
            #     reward = -0.1
            for tank_dict in self.enemy_last_seen:
                if tank_dict['position'][0] != -1000:
                    hide = False
                    line = [[self.x, self.y], [tank_dict['position'][0], tank_dict['position'][1]]]
                    for rect in obstacle_rects:
                        if line_segment_rect_intersect(line, rect):
                            hide = True
                            break
                    hides.append(hide)
            if all(hides):
                reward = 1
            else:
                reward = -1

        elif mission_dict['type'] == 'attack':
            reward = -1
            min_angle_diff = 180
            for tank_dict in self.enemy_last_seen:
                if tank_dict['position'][0] != -1000:
                    angle_diff = aiming_deviation(self.x, self.y, self.aim_angle,
                                                  tank_dict['position'][0], tank_dict['position'][1])
                    if angle_diff <= min_angle_diff:
                        min_angle_diff = angle_diff

            reward -= (min_angle_diff / 180) ** 1 * 0.1
            min_dist = min([dist for dist in self.radar_distance if dist > 0] + [1000])
            if min_dist < 8:
                reward -= 1
            if self.aim_target_type == 4:
                reward = 10
        else:
            reward = 0

        return reward


class virtual_Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_rotated_points(self):
        # 这里只是简单地返回障碍物的四个角，没有考虑旋转
        # 如果障碍物可以旋转，你需要添加逻辑来计算旋转后的点
        return [[self.x, self.y], [self.x + self.width, self.y],
                [self.x + self.width, self.y + self.height], [self.x, self.y + self.height]]


# 障碍物类
class Obstacle:
    def __init__(self, x, y, width, height, headless, type='house'):
        self.rect = pygame.Rect(x, y, width, height)
        self.headless = headless
        self.type = type

        self.center = (x + width/2, y + height/2)

    def draw(self, screen):
        if not self.headless:
            pygame.draw.rect(screen, (128, 128, 128), self.rect)  # 灰色障碍物
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)  # 绘制黑色边框

    # def get_rect(self):
    #     return self.rect

    def get_rotated_points(self):
        # 这里只是简单地返回障碍物的四个角，没有考虑旋转
        # 如果障碍物可以旋转，你需要添加逻辑来计算旋转后的点
        return [self.rect.topleft, self.rect.topright, self.rect.bottomright, self.rect.bottomleft]
