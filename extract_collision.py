#!/usr/bin/env python

import rospy
import pcl
import numpy as np
import math
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from moveit_commander import PlanningSceneInterface
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from pcl import pcl_visualization


class CylinderSegment:
    def __init__(self):
        rospy.init_node('cylinder_segment', anonymous=True)
        self.planning_scene_interface = PlanningSceneInterface()
        self.cloud_subscriber = rospy.Subscriber(
            "/camera/depth/color/points",
            PointCloud2,
            self.cloud_cb,
            queue_size=1,
            buff_size=2 ** 24  # 增加缓冲区大小处理大点云
        )
        self.cylinder_params = {
            'radius': 0,
            'direction_vec': np.array([0, 0, 0]),
            'center_pt': np.array([0, 0, 0]),
            'height': 0
        }
        rospy.loginfo("Cylinder segmentation node initialized")

    def ros_to_pcl(self, ros_cloud):
        """Convert ROS PointCloud2 to PCL PointCloud"""
        points = []
        for data in pc2.read_points(ros_cloud, skip_nans=True, field_names=("x", "y", "z")):
            points.append([data[0], data[1], data[2]])

        cloud = pcl.PointCloud()
        cloud.from_list(np.array(points, dtype=np.float32))
        return cloud

    def pass_through_filter(self, cloud):
        """Filter point cloud along Z-axis"""
        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name("z")
        passthrough.set_filter_limits(0.3, 1.1)
        return passthrough.filter()

    def compute_normals(self, cloud):
        """Estimate point normals"""
        ne = cloud.make_NormalEstimation()
        tree = cloud.make_kdtree()
        ne.set_SearchMethod(tree)
        ne.set_KSearch(50)
        return ne.compute()

    def remove_plane_surface(self, cloud):
        """Remove dominant plane using RANSAC"""
        seg = cloud.make_segmenter()
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_MaxIterations(1000)  # 注意这里使用大写M
        seg.set_distance_threshold(0.01)

        inliers, coefficients = seg.segment()

        extract = cloud.make_extract_indices()
        extract.set_negative(True)
        extract.set_indices(inliers)
        return extract.filter()

    def extract_cylinder(self, cloud, normals):
        """Extract cylinder using SAC segmentation"""
        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_CYLINDER)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_normal_distance_weight(0.1)
        seg.set_MaxIterations(10000)  # 圆柱检测需要更多迭代
        seg.set_distance_threshold(0.02)  # 增大阈值提高鲁棒性
        seg.set_radius_limits(0.01, 0.1)
        seg.set_input_normals(normals)

        inliers, coefficients = seg.segment()

        extract = cloud.make_extract_indices()
        extract.set_negative(False)
        extract.set_indices(inliers)
        return extract.filter(), coefficients

    def calculate_cylinder_pose(self, coefficients):
        """Calculate cylinder pose from coefficients"""
        # 圆柱轴线方向向量
        direction = np.array(coefficients[3:6])
        direction_norm = direction / np.linalg.norm(direction)

        # 计算与Z轴的旋转
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, direction_norm)

        if np.linalg.norm(rotation_axis) < 1e-6:
            # 方向平行于Z轴
            return Pose(), False

        rotation_axis /= np.linalg.norm(rotation_axis)
        rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))

        pose = Pose()
        pose.position.x = coefficients[0]
        pose.position.y = coefficients[1]
        pose.position.z = coefficients[2]
        pose.orientation.x = rotation_axis[0] * np.sin(rotation_angle / 2)
        pose.orientation.y = rotation_axis[1] * np.sin(rotation_angle / 2)
        pose.orientation.z = rotation_axis[2] * np.sin(rotation_angle / 2)
        pose.orientation.w = np.cos(rotation_angle / 2)

        return pose, True

    def add_cylinder_to_scene(self, radius, height, pose):
        """Add cylinder collision object to planning scene"""
        collision_object = CollisionObject()
        collision_object.header.frame_id = "camera_rgb_optical_frame"
        collision_object.id = "detected_cylinder"

        primitive = SolidPrimitive()
        primitive.type = primitive.CYLINDER
        primitive.dimensions = [height, radius]

        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = collision_object.ADD

        self.planning_scene_interface.apply_collision_object(collision_object)
        rospy.loginfo("Added cylinder to planning scene (r={}, h={})".format(radius, height))

    def cloud_cb(self, msg):
        """Main point cloud processing callback"""
        try:
            # 1. Convert to PCL
            cloud = self.ros_to_pcl(msg)
            if cloud.size == 0:
                rospy.logwarn("Empty cloud received")
                return

            # 2. Apply passthrough filter
            cloud = self.pass_through_filter(cloud)

            # 3. Estimate normals
            normals = self.compute_normals(cloud)

            # 4. Remove table plane
            cloud = self.remove_plane_surface(cloud)
            if cloud.size == 0:
                rospy.logwarn("No points left after plane removal")
                return

            # 5. Extract cylinder
            cloud, coefficients = self.extract_cylinder(cloud, normals)
            if cloud.size == 0:
                rospy.logwarn("No cylinder found")
                return

            # 6. Calculate and add to scene
            pose, success = self.calculate_cylinder_pose(coefficients)
            if success:
                self.add_cylinder_to_scene(
                    radius=coefficients[6],
                    height=self.estimate_cylinder_height(cloud),
                    pose=pose
                )

        except Exception as e:
            rospy.logerr("Point cloud processing failed: {}".format(str(e)))

    def estimate_cylinder_height(self, cloud):
        """Estimate cylinder height from point cloud"""
        points = np.asarray(cloud)
        if len(points) == 0:
            return 0.1  # default height

        # Project points onto cylinder axis
        z_values = points[:, 2]  # 假设圆柱大致沿Z轴方向
        return np.max(z_values) - np.min(z_values)


if __name__ == '__main__':
    try:
        segmenter = CylinderSegment()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass