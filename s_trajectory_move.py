import os
if "ROS_NAMESPACE" not in os.environ:
    os.environ["ROS_NAMESPACE"] = "arm"
import rospy
import math
from geometry_msgs.msg import Pose
import sys
import moveit_commander

if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('s_shape_trajectory')

    # ???MoveIt??
    robot = moveit_commander.RobotCommander()
    group_name = "manipulator"
    # eef_link = "camera_virtual_link"
    # eef_link = "fork_grappling_link"

    move_group = moveit_commander.MoveGroupCommander(group_name)
    scene = moveit_commander.PlanningSceneInterface(synchronous=True)

    # 获取当前末端位姿
    planning_frame = move_group.get_planning_frame()
    eff_link = move_group.get_end_effector_link()
    start_pose = move_group.get_current_pose().pose

    # 定义S形轨迹参数
    amplitude = 0.3  # S形幅度
    length = 0.5  # S形长度
    steps = 30  # 轨迹点数

    waypoints = []

    # 生成S形轨迹点
    for i in range(steps + 1):
        # x = start_pose.position.x + (i / steps) * length
        # y = start_pose.position.y + amplitude * math.sin(2 * math.pi * (i / steps))
        # z = start_pose.position.z

        '''y-z平面'''
        x = start_pose.position.x
        y = start_pose.position.y + amplitude * math.sin(2 * math.pi * (i / steps))
        z = start_pose.position.z + (i / steps) * length
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = start_pose.orientation

        waypoints.append(pose)

    # 计算笛卡尔路径
    (plan, fraction) = move_group.compute_cartesian_path(
        waypoints,  # 路径点
        0.01,  # eef_step
    )

    # 执行轨迹
    if fraction > 0.9:  # 至少完成了90%的路径
        move_group.execute(plan, wait=True)
    else:
        rospy.logwarn("Failed to compute complete path (%.2f%%)", fraction * 100)