import numpy as np
import copy
import ruamel.yaml


def trans_mat_3x3_from_array(cord):
    x, y, theta = [float(value) for value in cord]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def is_pose_empty(pose: dict):
    return any(key_value == '' for key_value in pose.values())


def is_too_close(pose_1: dict, pose_2: dict):
    threshold = 3
    for side, pose in pose_1.items():
        x_1 = float(pose['poseX'])
        y_1 = float(pose['poseY'])
        for other_side, other_pose in pose_2.items():
            x_2 = float(other_pose['poseX'])
            y_2 = float(other_pose['poseY'])
            if np.linalg.norm([x_1 - x_2, y_1 - y_2]) < threshold:
                return True
    return False


def add_pose_to_parent(code_num: str, code_pose: dict, parent: dict):
    if code_num not in parent:
        parent[code_num] = {}
    for side, pose in code_pose.items():
        if side not in parent[code_num]:
            parent[code_num][side] = copy.deepcopy(pose)


def compute_relative_pose(ref_pose: dict, child_pose: dict):
    relative_target_trans_mat = {}
    for side, side_pose in ref_pose.items():
        x_0 = side_pose['poseX']
        y_0 = side_pose['poseY']
        theta_0 = side_pose['angle']
    ref_trans_mat = trans_mat_3x3_from_array([float(x_0), float(y_0), float(theta_0)])
    for side, pose in child_pose.items():
        child_trans_mat = trans_mat_3x3_from_array([pose['poseX'], pose['poseY'], pose['angle']])
        relative_trans_mat = np.dot(np.linalg.inv(ref_trans_mat), child_trans_mat)
        relative_target_trans_mat[side] = relative_trans_mat
    return relative_target_trans_mat


def extract_coordinates(transform_matrix):
    # 提取平移距离
    x = transform_matrix[0, 2]
    y = transform_matrix[1, 2]
    # 提取旋转角度
    theta = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
    return x, y, theta


def compute_new_pose(anchor_pose: dict, trans_mat_3x3_target_wrt_ref):
    side_key = next(iter(anchor_pose))
    anchor_trans_mat = trans_mat_3x3_from_array(
        [anchor_pose[side_key]['poseX'], anchor_pose[side_key]['poseY'], anchor_pose[side_key]['angle']])
    trans_mat_3x3_target_wrt_new_map = np.dot(anchor_trans_mat, trans_mat_3x3_target_wrt_ref)
    return extract_coordinates(trans_mat_3x3_target_wrt_new_map)


class PoseProcessor:
    def __init__(self, template_data_path, new_anchor_data_path, new_data_path):
        self.yaml = ruamel.yaml.YAML()
        self.yaml.preserve_quotes = True
        self.yaml.typ = "safe"
        self.template_data_path = template_data_path
        self.new_anchor_data_path = new_anchor_data_path
        self.yaml_new_data_path = new_data_path
        self.data = None
        self.filter_data = {}
        self.anchor_points = None
        self.par_child = None
        self.new_anchor_points = {}
        self.new_filter_data = {}

    def load_data(self, file_path):
        with open(file_path, "r") as file:
            data = self.yaml.load(file)
        return data

    def set_new_anchor_points(self):
        self.new_anchor_points = copy.deepcopy(self.anchor_points)
        '''load new anchor points'''
        new_anchor_data = self.load_data(file_path=self.new_anchor_data_path)
        '''update new anchor points'''
        for anchor_num, anchor_pose in self.anchor_points.items():
            print(f"anchor_num: {anchor_num}, old anchor_pose:{anchor_pose}")
            side = list(anchor_pose.keys())[0]
            self.new_anchor_points[anchor_num][side] = new_anchor_data[anchor_num][side]
            new_anchor_pose = self.new_anchor_points[anchor_num]
            print(f"anchor_num: {anchor_num}, new anchor_pose:{new_anchor_pose}")

            print("-----")

    def set_new_filter_data(self):
        self.new_filter_data = copy.deepcopy(self.filter_data)

    def filter_empty_pose(self):
        for code_num, code_pose in self.data.items():
            for side, pose in code_pose.items():
                if not is_pose_empty(pose):  # 假设 is_pose_empty 正确判断姿态是否为空
                    # 只有在pose不为空时，才添加到filter_data中
                    if code_num not in self.filter_data:
                        self.filter_data[code_num] = {}  # 初始化code_num对应的字典
                    self.filter_data[code_num][side] = pose  # 添加或更新side对应的姿态数据

    def select_anchor_points(self):
        self.anchor_points = {}
        self.par_child = {}
        for code_num, code_pose in self.filter_data.items():
            # 检查当前code_num是否与anchor_num中的任何元素匹配
            add = True
            for anchor_code_num in self.anchor_points.keys():
                if 'BK' in code_num:  # and 'BK' in anchor_code_num
                    if code_num.startswith(anchor_code_num[:5]) or is_too_close(code_pose,
                                                                                self.anchor_points[anchor_code_num]):
                        add_pose_to_parent(code_num, code_pose, self.par_child[anchor_code_num])
                        add = False
                        break
                else:
                    if code_num.split('_')[0] == anchor_code_num.split('_')[0] or is_too_close(code_pose,
                                                                                               self.anchor_points[
                                                                                                   anchor_code_num]):
                        add_pose_to_parent(code_num, code_pose, self.par_child[anchor_code_num])

                        add = False
                        break
            # 如果add标志仍然为True，说明当前code_num没有匹配，可以添加到anchor_num
            if add:
                if code_num not in self.anchor_points:
                    self.anchor_points[code_num] = {}
                    self.par_child[code_num] = {}
                    add_pose_to_parent(code_num, code_pose, self.par_child[code_num])

                side, pose = copy.deepcopy(code_pose).popitem()
                self.anchor_points[code_num][side] = pose

        for key in self.anchor_points:
            if key not in self.par_child:
                raise Exception('dimension not match')

    def save_data(self, data):
        with open(self.yaml_new_data_path, "w") as file:
            self.yaml.dump(data, file)

    def pre_process(self):
        self.data = self.load_data(file_path=self.template_data_path)  # 加载原始数据
        self.filter_empty_pose()  # 过滤掉包含空姿态的数据
        self.select_anchor_points()  # 选择锚点

    def transfer_coord(self):
        # 假设我们已经有了过滤和锚点数据，现在进行相对姿态和新姿态的计算
        for anchor_num, anchor_pose in self.anchor_points.items():
            for child_name, child_pose in self.par_child[anchor_num].items():
                # 计算相对姿态
                relative_trans_mat = compute_relative_pose(anchor_pose, child_pose)
                self.par_child[anchor_num][child_name] = relative_trans_mat  # 更新相对姿态数据

        # 计算新的姿态并更新过滤数据
        # self.set_new_anchor_points()
        self.set_new_filter_data()
        for anchor_num, anchor_pose in self.new_anchor_points.items():
            for child_name, child_pose in self.par_child[anchor_num].items():
                for side, trans_mat_3x3 in child_pose.items():
                    new_pose = compute_new_pose(anchor_pose, trans_mat_3x3)
                    self.new_filter_data[child_name][side]['poseX'] = str(new_pose[0])
                    self.new_filter_data[child_name][side]['poseY'] = str(new_pose[1])
                    self.new_filter_data[child_name][side]['angle'] = str(new_pose[2])

        # 保存更新后的数据到 YAML 文件
        self.save_data(self.new_filter_data)

    def print_anchor_points(self):
        for anchor_num, anchor_pose in self.anchor_points.items():
            print(f"anchor_num: {anchor_num}, anchor_pose:{anchor_pose}")
            print("-----")

        print("-----------------------------------------")

    def print_par_child(self):
        for parent, children in self.par_child.items():
            print(f"parent:{parent}")
            print("-----")
            print(f"children: {children}")
            print("-----------------")

# 使用类
pose_processor = PoseProcessor(template_data_path='/home/gr-arm-7xx2/Downloads/data.yaml',
                               new_anchor_data_path='/home/gr-arm-7xx2/Downloads/data_05_11.yaml',
                               new_data_path="/home/gr-arm-7xx2/Downloads/data_new.yaml")
pose_processor.pre_process()
'''check the anchor points'''
pose_processor.print_anchor_points()
pose_processor.print_par_child()
pose_processor.set_new_anchor_points()  # modify coordinates of anchor points every time
pose_processor.transfer_coord()
