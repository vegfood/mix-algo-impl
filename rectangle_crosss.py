import math
import time

import numpy as np
from typing import Tuple, List


def dot(v1, v2):
    """计算两个向量的点积"""
    return v1[0] * v2[0] + v1[1] * v2[1]


def project(poly, axis):
    """将多边形投影到轴上，返回投影的最小值和最大值"""
    min_proj = max_proj = dot(poly[0], axis)
    for point in poly[1:]:
        proj = dot(point, axis)
        min_proj = min(min_proj, proj)
        max_proj = max(max_proj, proj)
    return (min_proj, max_proj)


def overlaps(proj1, proj2):
    """检查两个投影是否重叠"""
    return not (proj1[1] < proj2[0] or proj2[1] < proj1[0])


def get_edges(poly):
    """获取多边形的边向量"""
    edges = []
    n = len(poly)
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        edges.append((p2[0] - p1[0], p2[1] - p1[1]))
    return edges


def get_axes(edges):
    """从边向量获取分离轴（法向量）"""
    axes = []
    for edge in edges:
        # 法向量是边的垂直向量
        normal = (-edge[1], edge[0])
        # 归一化（可选，但有助于数值稳定性）
        length = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if length > 0:
            normal = (normal[0] / length, normal[1] / length)
        axes.append(normal)
    return axes


def polygons_intersect(poly1, poly2):
    """
    判断两个凸多边形是否相交
    参数:
        poly1: 第一个多边形的顶点列表 [(x1,y1), (x2,y2), ...]
        poly2: 第二个多边形的顶点列表 [(x1,y1), (x2,y2), ...]
    返回:
        bool: 是否相交
    """
    # 获取所有可能的分离轴
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)
    axes = get_axes(edges1) + get_axes(edges2)

    # 检查每个分离轴
    for axis in axes:
        if axis == (0, 0):
            continue  # 忽略零向量

        # 投影两个多边形
        proj1 = project(poly1, axis)
        proj2 = project(poly2, axis)

        # 如果发现不重叠的投影，则不相交
        if not overlaps(proj1, proj2):
            return False

    return True


# 矩形相交的便捷函数
def rectangles_intersect(rect1, rect2, is_axis_aligned=False):
    """
    判断两个矩形是否相交
    参数:
        rect1: 第一个矩形的表示
            - 如果是轴对齐矩形: (x_min, y_min, x_max, y_max)
            - 如果是旋转矩形: 四个顶点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] 顺时针或逆时针顺序
        rect2: 第二个矩形的表示（格式同rect1）
        is_axis_aligned: 是否为轴对齐矩形
    返回:
        bool: 是否相交
    """
    if is_axis_aligned:
        # 轴对齐矩形的快速判断
        x_min1, y_min1, x_max1, y_max1 = rect1
        x_min2, y_min2, x_max2, y_max2 = rect2

        # 检查x和y投影是否重叠
        if x_max1 < x_min2 or x_max2 < x_min1:
            return False
        if y_max1 < y_min2 or y_max2 < y_min1:
            return False
        return True
    else:
        # 旋转矩形的通用判断
        return polygons_intersect(rect1, rect2)


def generate_detect_area(vel: List[float]):
    """
    :param vel: List of 3 float values representing velocity components in x, y, and z directions.
    :return: Tuple containing width, length, reference center x position, and reference center y position for the detection area.
    """
    # 检查输入数组的长度是否为3
    if len(vel) != 3:
        raise ValueError("vel must be a list of 3 floats")
    reaction_time = 1
    deceleration = 0.5  # 0.5 m/s^2
    length = 1  # 车长

    if np.linalg.norm(vel) <= 1e-3:
        # 停止
        width = 0.8
        length = 1
        ref_center_x = 0.2
        ref_center_y = 0

    elif np.linalg.norm(vel[0:2]) > 3e-2:
        # 前进
        width = 0.8
        d1 = np.linalg.norm(vel[0:2]) * reaction_time
        d2 = (vel[0] ** 2 + vel[1] ** 2) / (2 * deceleration)
        length += (d1 + d2)
        ref_center_x = - ((length / 2) - 0.7)
        ref_center_y = 0

    else:
        # 旋转
        width = 1.2
        length = 1.2
        ref_center_x = 0
        ref_center_y = 0

    return width, length, ref_center_x, ref_center_y


def generate_collision_rectangle(cx, cy, width, length, angle_rads, ref_center_x=0.0, ref_center_y=0.0):
    """
    生成碰撞矩形的全局顶点坐标（支持非中心参考点）

    参数:
        cx, cy:         AGV当前中心坐标
        width:          矩形短边（宽度）
        length:         矩形长边（长度）
        angle_rads:  旋转角度（0朝向x轴正方向，逆时针旋转）
        ref_center_x/y: 可选参数，定义局部坐标系中的参考中心点（默认 None 表示几何中心）

    返回:
        list: 四个全局顶点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]， 后右，前右，前左，后左
    """
    half_width = width / 2
    half_length = length / 2

    # 局部坐标系顶点（基于几何中
    vertices_local = [
        (-half_length, -half_width),  # 后右
        (half_length, -half_width),  # 前右
        (half_length, +half_width),  # 前左
        (-half_length, half_width),  # 后左
    ]

    # 如果指定了非中心参考点，调整局部坐标
    dx = -ref_center_x  # 参考点到几何中心的偏移，表示在局部坐标系
    dy = -ref_center_y
    vertices_local = [(x + dx, y + dy) for x, y in vertices_local]

    # 旋转和平移到全局坐标系
    cos_val = math.cos(angle_rads)
    sin_val = math.sin(angle_rads)

    vertices_global = []
    for x, y in vertices_local:
        x_rot = x * cos_val - y * sin_val + cx
        y_rot = x * sin_val + y * cos_val + cy
        vertices_global.append((x_rot, y_rot))

    return vertices_global


# 示例使用
if __name__ == "__main__":
    t = time.time()
    v1 = [0, 1, 0.5]
    v2 = [1, 0.0, 0.5]

    # if np.linalg.norm(v) <= 1e-5:
    #     # 停止
    #     width = 0.8
    #     length = 1
    #     ref_center_x = 0.2
    #     ref_center_y = 0
    #
    # elif np.linalg.norm(v[0:2]) > 3e-2:
    #     # 前进
    #     width = 0.8
    #     length = 3
    #     ref_center_x = -0.8
    #     ref_center_y = 0
    # else:
    #     # 旋转
    #     width = 1.2
    #     length = 1.2
    #     ref_center_x = 0
    #     ref_center_y = 0
    width1, length1, ref_center_x1, ref_center_y1 = generate_detect_area(v1)
    width2, length2, ref_center_x2, ref_center_y2 = generate_detect_area(v2)
    # 案例1：基于非中心参考点（如AGV的尾部中心）
    rect1 = generate_collision_rectangle(cx=2, cy=0, width=width1, length=length1, angle_rads=1.57,
                                         ref_center_x=ref_center_x1, ref_center_y=ref_center_y1)
    print("基于几何中心的旋转:")
    for i, (x, y) in enumerate(rect1, 1):
        print(f"顶点{i}: ({x:.2f}, {y:.2f})")

    # 案例2：基于非中心参考点（如AGV的尾部中心）
    rect2 = generate_collision_rectangle(cx=0, cy=2, width=width2, length=length2, angle_rads=0,
                                         ref_center_x=ref_center_x2, ref_center_y=ref_center_y2)
    print("\n基于尾部中心的旋转:")
    for i, (x, y) in enumerate(rect2, 1):
        print(f"顶点{i}: ({x:.2f}, {y:.2f})")

    print(rectangles_intersect(rect1, rect2))  # 输出: False
    print(time.time() - t)

    # 旋转矩形表示为四个顶点（顺时针或逆时针顺序）
    # 矩形1: 未旋转的矩形 (0,0)-(2,0)-(2,2)-(0,2)
    rect1 = [(0, 0), (2, 0), (2, 2), (0, 2)]

    # 矩形2: 旋转45度的矩形 (中心在(2,2)，边长为2√2)
    rect2 = [(1, 1), (3, 1), (3, 3), (1, 3)]

    # print(rectangles_intersect(rect1, rect2))  # 输出: True
    # print(time.time() - t)

    # 矩形1: (0,0)-(1,0)-(1,1)-(0,1)
    rect1 = [(0, 0), (1, 0), (1, 1), (0, 1)]

    # 矩形2: (2,2)-(3,2)-(3,3)-(2,3)
    rect2 = [(2, 2), (3, 2), (3, 3), (2, 3)]

    # print(rectangles_intersect(rect1, rect2))  # 输出: False
