import math
import numpy as np


def find_closest_point_on_line_segment(p1, p2, p0):
    """
    计算点p0到线段p1-p2的最近点
    :param p1: 线段起点
    :param p2: 线段终点
    :param p0: 外部点
    :return: 最近点坐标
    """
    line_vec = p2 - p1
    point_vec = p0 - p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    if t < 0.0:
        return p1
    elif t > 1.0:
        return p2
    projection = p1 + t * line_vec
    return projection


def find_closest_point_to_polygon(polygon, point):
    """
    计算点到多边形最近的点
    :param polygon: 多边形顶点列表
    :param point: 外部点
    :return: 最近点坐标
    """
    closest_point = None
    min_distance = np.inf
    for i in range(len(polygon)):
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[(i + 1) % len(polygon)])
        closest_point_on_edge = find_closest_point_on_line_segment(p1, p2, point)
        distance = np.linalg.norm(point - closest_point_on_edge)
        if distance < min_distance:
            min_distance = distance
            closest_point = closest_point_on_edge
    return closest_point


def line_intersects_line(line1_start, line1_end, line2_start, line2_end):
    """检测两线段是否相交"""
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return False  # 平行或共线

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    return 0 <= t <= 1 and 0 <= u <= 1


def find_intersection_point(line1_start, line1_end, line2_start, line2_end):
    """计算两线段的交点"""
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行或共线

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    return x1 + t * (x2 - x1), y1 + t * (y2 - y1)


def get_rect_edges(rect):
    """获取矩形的四条边"""
    return [
        (rect.topleft, rect.topright),
        (rect.bottomleft, rect.bottomright),
        (rect.topleft, rect.bottomleft),
        (rect.topright, rect.bottomright)
    ]


def find_first_intersection(line_start, line_end, rects):
    """找到线段与一系列矩形的第一个交点"""
    closest_intersection = None
    closest_distance = float('inf')

    for rect in rects:
        for edge_start, edge_end in get_rect_edges(rect):
            if line_intersects_line(line_start, line_end, edge_start, edge_end):
                intersection = find_intersection_point(line_start, line_end, edge_start, edge_end)
                if intersection:
                    distance = math.hypot(intersection[0] - line_start[0], intersection[1] - line_start[1])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_intersection = intersection

    return closest_intersection


def rotate_point(x, y, angle, cx, cy):
    """ 旋转点围绕中心(cx, cy) """
    radians = math.radians(angle)
    xx = cx + math.cos(radians) * (x - cx) - math.sin(radians) * (y - cy)
    yy = cy + math.sin(radians) * (x - cx) + math.cos(radians) * (y - cy)
    return xx, yy


# 碰撞检测函数
# def check_collision(tank1, tank2):
#     # 这里我们可以实现更复杂的多边形碰撞检测
#     # 简单起见，仍然使用Rect碰撞检测作为示例
#     return tank1.get_rect().colliderect(tank2.get_rect())


# 多边形和点的碰撞检测
def point_inside_polygon(point, polygon):
    """使用射线法判断点是否在多边形内部"""
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


# 多边形和多边形的碰撞检测
def polygon_collision(poly1, poly2):
    """检查两个多边形是否相交"""
    for point in poly1:
        if point_inside_polygon(point, poly2):
            return True
    for point in poly2:
        if point_inside_polygon(point, poly1):
            return True
    return False


# 使用多边形碰撞检测的函数
def polygon_line_collision(polygon, line_start, line_end):
    """检查多边形和线段是否相交"""
    for i in range(len(polygon)):
        segment_start = polygon[i]
        segment_end = polygon[(i + 1) % len(polygon)]
        if line_intersects_line(segment_start, segment_end, line_start, line_end):
            return True
    return False


def calculate_polygon_line_intersection(polygon, line_start, line_end):
    """计算多边形和线段的交点"""
    closest_intersection = None
    closest_distance = float('inf')

    for i in range(len(polygon)):
        poly_start = polygon[i]
        poly_end = polygon[(i + 1) % len(polygon)]

        intersection = get_line_intersection(poly_start, poly_end, line_start, line_end)
        if intersection:
            distance = math.hypot(intersection[0] - line_start[0], intersection[1] - line_start[1])
            if distance < closest_distance:
                closest_distance = distance
                closest_intersection = intersection

    return closest_intersection


def get_line_intersection(p0, p1, p2, p3):
    """计算两线段的交点，如果没有交点返回None"""
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y
    if denom == 0:  # 平行或共线
        return None

    denom_is_positive = denom > 0

    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]

    s_numer = s10_x * s02_y - s10_y * s02_x
    if (s_numer < 0) == denom_is_positive:
        return None

    t_numer = s32_x * s02_y - s32_y * s02_x
    if (t_numer < 0) == denom_is_positive:
        return None

    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive:
        return None

    # 线段有交点
    t = t_numer / denom
    intersection_point = (p0[0] + (t * s10_x), p0[1] + (t * s10_y))

    return intersection_point


def is_overlap(rect1, rect2):
    """
    检查两个矩形是否重叠。
    每个矩形通过一个元组表示: (x中心, y中心, 宽度, 高度)
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # 检查矩形1的任意一边是否超出矩形2的对应边
    if (x1 + w1 / 2 < x2 - w2 / 2) or (x1 - w1 / 2 > x2 + w2 / 2) or (y1 + h1 / 2 < y2 - h2 / 2) or (
            y1 - h1 / 2 > y2 + h2 / 2):
        return False  # 没有重叠
    return True  # 重叠


def rotate_point2(cx, cy, angle, px, py):
    """
    旋转点(px, py)围绕中心点(cx, cy)角度angle。
    angle以弧度为单位。
    """
    s = np.sin(angle)
    c = np.cos(angle)

    # translate point back to origin:
    px -= cx
    py -= cy

    # rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # translate point back:
    px = xnew + cx
    py = ynew + cy
    return px, py


def get_rect_points(cx, cy, w, h, angle):
    """
    获取旋转后的矩形的四个角点。
    (cx, cy)是矩形中心，w和h是宽度和高度，angle是旋转角度（弧度）。
    """
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    rotated_corners = [rotate_point2(0, 0, angle, x, y) for x, y in corners]
    rotated_corners = [(cx + x, cy + y) for x, y in rotated_corners]
    return rotated_corners


def project_polygon(axis, polygon):
    """
    将多边形投影到轴上。
    """
    dots = [np.dot(axis, point) for point in polygon]
    return min(dots), max(dots)


def overlapping(a_min, a_max, b_min, b_max):
    """
    检查两个投影是否重叠。
    """
    return a_min <= b_max and b_min <= a_max


def is_separating_axis(axis, rect1, rect2):
    """
    检查给定轴是否为分离轴。
    """
    proj1_min, proj1_max = project_polygon(axis, rect1)
    proj2_min, proj2_max = project_polygon(axis, rect2)
    return not overlapping(proj1_min, proj1_max, proj2_min, proj2_max)


def rectangles_overlap(rect1, rect2):
    """
    检查两个旋转矩形是否重叠。
    rect1和rect2是旋转矩形的顶点列表。
    """
    # 为两个矩形的每条边找到法线作为分离轴
    axes = []
    for rect in [rect1, rect2]:
        for i in range(len(rect)):
            p1, p2 = rect[i], rect[(i + 1) % len(rect)]
            edge = np.subtract(p2, p1)
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal / np.linalg.norm(normal))

    # 检查每个轴是否为分离轴
    for axis in axes:
        if is_separating_axis(axis, rect1, rect2):
            return False  # 找到分离轴，矩形不重叠

    return True  # 未找到分离轴，矩形重叠


def cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_segment_intersect(p1, p2, p3, p4):
    """
    判断两条线段 p1p2 和 p3p4 是否相交
    :param p1, p2: 线段1的两个端点
    :param p3, p4: 线段2的两个端点
    :return: 布尔值，表示两条线段是否相交
    """
    # 计算向量
    d1 = (p2[0] - p1[0], p2[1] - p1[1])  # 线段p1p2的方向向量
    d2 = (p4[0] - p3[0], p4[1] - p3[1])  # 线段p3p4的方向向量
    # 计算叉积
    cp1 = cross_product((p3[0] - p1[0], p3[1] - p1[1]), d1)
    cp2 = cross_product((p4[0] - p1[0], p4[1] - p1[1]), d1)
    cp3 = cross_product((p1[0] - p3[0], p1[1] - p3[1]), d2)
    cp4 = cross_product((p2[0] - p3[0], p2[1] - p3[1]), d2)

    # 判断两条线段是否相交
    return (cp1 * cp2 <= 0) and (cp3 * cp4 <= 0)


def line_segment_rect_intersect(line, rect):
    """
    检测线段与矩形是否相交
    """
    edges = [[rect[0], rect[1]], [rect[1], rect[2]], [rect[2], rect[3]], [rect[3], rect[0]]]
    for edge in edges:
        if line_segment_intersect(line[0], line[1], edge[0], edge[1]):
            return True
    else:
        return False


def aiming_deviation(x1, y1, aim_angle, x2, y2):
    """
    计算tank2相对于tank1的瞄准角度偏差，angle是tank1的aim_angle参数
    """
    angle_to_enemy = math.atan2(y2 - y1, x2 - x1)
    angle_to_enemy = math.degrees(angle_to_enemy) % 360
    turret_angle = aim_angle % 360
    # 计算夹角
    return min(abs(turret_angle - angle_to_enemy), 360 - abs(turret_angle - angle_to_enemy))


def angle_difference(angle1, angle2):
    # 计算两个角度的差值
    diff = angle2 - angle1
    # 调整差值使其在-180到180度范围内
    if diff > 180:
        diff -= 360
    elif diff <= -180:
        diff += 360
    return diff


if __name__ == '__main__':
    rect1 = [[0, 0], [1, 0], [1, 1], [0, 1]]
    rect2 = [[2, 2], [3, 2], [3, 3], [2, 3]]
    print(rectangles_overlap(rect1, rect2))
