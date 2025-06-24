#!/usr/bin/env python
import os
if "ROS_NAMESPACE" not in os.environ:
    os.environ["ROS_NAMESPACE"] = "arm"
import rospy
import math
from moveit_commander import MoveGroupCommander
from tf.transformations import quaternion_from_euler
import threading
from geometry_msgs.msg import Pose, Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

class YZPlaneZigZag:
    def __init__(self):
        rospy.init_node('yz_zigzag_generator')
        self.group = MoveGroupCommander("manipulator")  # 替换为实际group名称

        # 运动参数配置
        self.config = {
            'z_start': 0.6,  # Z轴起始位置(m)
            'y_amplitude': 0.5,  # Y轴摆动幅度(m)
            'z_step': 0.04,  # Z轴步进长度(m)
            'num_pass': 8,  # 完整往复次数
            'points_per_pass': 10,  # 单次往返轨迹点数
            'speed_factor': 0.6,  # 速度缩放因子
            'accel_factor': 0.4  # 加速度缩放因子
        }

        # 初始化运动规划器
        self.group.set_max_velocity_scaling_factor(self.config['speed_factor'])
        self.group.set_max_acceleration_scaling_factor(self.config['accel_factor'])
        self.group.allow_replanning(True)

        # 获取初始姿态
        self.start_pose = self.group.get_current_pose().pose

        # 轨迹可视化参数
        self.marker_scale = 0.02  # 轨迹点标记尺寸(m)
        self.traj_history_len = 200  # 最大历史轨迹点数
        self.plan_color = (0.0, 1.0, 0.0, 0.8)  # 规划轨迹颜色(RGBA)
        self.real_color = (1.0, 0.0, 0.0, 0.6)  # 实际轨迹颜色

        # 初始化可视化发布器
        self.marker_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
        self.traj_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=20)

        # 轨迹记录容器
        self.planned_traj = []
        self.real_traj = []
        self.lock = threading.Lock()

        # 启动实时轨迹记录线程
        self.recording_thread = threading.Thread(target=self.record_real_trajectory)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def create_marker(self, position, color, ns, id, type=Marker.SPHERE):
        """创建可视化标记"""
        marker = Marker()
        marker.header.frame_id = self.group.get_planning_frame()
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = type
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(self.marker_scale, self.marker_scale, self.marker_scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def publish_planned_trajectory(self, waypoints):
        """发布规划轨迹的可视化标记"""
        marker_array = MarkerArray()

        # 删除旧标记
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 生成新标记
        for i, pose in enumerate(waypoints):
            marker = self.create_marker(
                position=pose.position,
                color=self.plan_color,
                ns="planned_traj",
                id=i
            )
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)
        self.planned_traj = waypoints.copy()

    def record_real_trajectory(self):
        """实时记录并发布实际运动轨迹"""
        rate = rospy.Rate(30)  # 30Hz采样率
        while not rospy.is_shutdown():
            try:
                current_pose = self.group.get_current_pose().pose
                with self.lock:
                    self.real_traj.append(current_pose)
                    # 保持队列长度
                    if len(self.real_traj) > self.traj_history_len:
                        self.real_traj.pop(0)
                self.publish_real_trajectory()
            except:
                rospy.logerr("Error getting current pose")
            rate.sleep()

    def publish_real_trajectory(self):
        """发布实际运动轨迹"""
        with self.lock:
            marker_array = MarkerArray()

            # 创建线标记
            line_marker = Marker()
            line_marker.header.frame_id = self.group.get_planning_frame()
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "real_traj_line"
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = self.marker_scale / 2
            # 正确设置颜色值
            if len(self.real_color) == 4:
                line_marker.color.r = self.real_color[0]
                line_marker.color.g = self.real_color[1]
                line_marker.color.b = self.real_color[2]
                line_marker.color.a = self.real_color[3]
            else:
                rospy.logwarn("Invalid color format, using default red")
                line_marker.color.r = 1.0
                line_marker.color.g = 0.0
                line_marker.color.b = 0.0
                line_marker.color.a = 1.0
            # line_marker.color.r, line_marker.color.g, line_marker.color.b, line_marker.a = self.real_color

            # 添加路径点
            for pose in self.real_traj:
                line_marker.points.append(pose.position)
                # 添加点标记
                point_marker = self.create_marker(
                    position=pose.position,
                    color=self.real_color,
                    ns="real_traj_points",
                    id=len(marker_array.markers)
                )
                marker_array.markers.append(point_marker)

            marker_array.markers.insert(0, line_marker)
            self.marker_pub.publish(marker_array)

    def show_moveit_trajectory(self, plan):
        """显示MoveIt规划轨迹"""
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.group.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.traj_pub.publish(display_trajectory)

    def generate_zigzag_points(self):
        """生成Y-Z平面之字形路径点"""
        waypoints = []
        current_z = self.config['z_start']
        direction = 1  # 方向标记：1正向，-1反向

        for _ in range(self.config['num_pass']):
            # Y轴摆动路径
            for i in range(self.config['points_per_pass'] + 1):
                t = i / self.config['points_per_pass']
                y = self.start_pose.position.y + self.config['y_amplitude'] * math.sin(math.pi * t) * direction
                z = current_z + (self.config['z_step'] / self.config['points_per_pass']) * i

                pose = Pose()
                pose.position = Point(self.start_pose.position.x, y, z)
                pose.orientation = self.start_pose.orientation
                waypoints.append(pose)

            current_z += self.config['z_step']
            direction *= -1  # 反转摆动方向

        return waypoints

    def optimize_trajectory(self, waypoints):
        """轨迹优化处理"""
        # 1. 简化路径点
        simplified_points = [waypoints[0]]
        for p in waypoints[1:]:
            if self._distance(simplified_points[-1], p) > 0.02:  # 间距大于2cm保留
                simplified_points.append(p)

        # 2. 平滑处理
        smoothed_points = []
        for i in range(1, len(simplified_points) - 1):
            # 三点平均滤波
            prev = simplified_points[i - 1].position
            curr = simplified_points[i].position
            next_p = simplified_points[i + 1].position

            smoothed = Pose()
            smoothed.position.x = (prev.x + curr.x + next_p.x) / 3
            smoothed.position.y = (prev.y + curr.y + next_p.y) / 3
            smoothed.position.z = (prev.z + curr.z + next_p.z) / 3
            smoothed.orientation = self.start_pose.orientation
            smoothed_points.append(smoothed)

        return simplified_points  # 返回优化后的路径点

    def _distance(self, pose1, pose2):
        """计算两个位姿间的欧氏距离"""
        p1 = pose1.position
        p2 = pose2.position
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def execute_trajectory(self):
        """执行完整轨迹流程"""
        rospy.loginfo("Generating Y-Z plane zigzag trajectory...")

        # 生成原始路径点
        raw_waypoints = self.generate_zigzag_points()
        rospy.loginfo(f"Generated {len(raw_waypoints)} raw waypoints")

        # 优化轨迹
        optimized_waypoints = self.optimize_trajectory(raw_waypoints)
        rospy.loginfo(f"Optimized to {len(optimized_waypoints)} points")

        # 显示规划轨迹
        # self.publish_planned_trajectory(raw_waypoints)

        # 笛卡尔路径规划
        (plan, fraction) = self.group.compute_cartesian_path(
            optimized_waypoints,
            eef_step=0.01,  # 1cm插值分辨率
            avoid_collisions=True)

        if fraction >= 0.85:
            # 显示MoveIt原始规划
            self.show_moveit_trajectory(plan)
            rospy.sleep(1)  # 等待显示更新
            rospy.loginfo(f"Executing trajectory ({fraction * 100:.1f}% coverage)")
            self.group.execute(plan, wait=True)
            rospy.loginfo("Trajectory execution completed!")
        else:
            rospy.logerr(f"Path planning failed! Only {fraction * 100:.1f}% coverage")

        # 返回初始位置
        # rospy.loginfo("Returning to start position...")
        # self.group.set_named_target("home")
        # self.group.go(wait=True)


if __name__ == '__main__':
    try:
        zigzag = YZPlaneZigZag()
        rospy.sleep(1)  # 等待初始化完成
        zigzag.execute_trajectory()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
