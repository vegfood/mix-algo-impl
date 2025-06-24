from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    Simulator, Meshcat, MeshcatVisualizer, FindResourceOrThrow,
    RigidTransform, MeshcatVisualizerParams, Role, Rgba,
)

# 创建系统框图
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

# 设置重力
plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.81])

# 加载机械臂URDF模型
# urdf_path = FindResourceOrThrow("package://iiwa_description/urdf/iiwa14_spheres_collision.urdf")
# urdf_path = FindResourceOrThrow("package://drake_models/iiwa_description/urdf/iiwa14_spheres_collision.urdf")
urdf_path = "/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/models/iiwa_description/urdf/iiwa14_spheres_collision.urdf"
model_root = "/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/models"

# with open(urdf_path, 'r') as f:
#     print(f.read())  # 确认是否存在 <mass> 和 <inertia> 标签

parser = Parser(plant)
# parser.package_map().Add("iiwa_description", f"{model_root}/iiwa_description")

# arm_model = parser.AddModels(urdf_path)[0]
# arm_model = parser.AddModelsFromUrl(f"file://{urdf_path}")[0]
# 使用package://协议加载
urdf_url = "package://drake_models/iiwa_description/urdf/iiwa14_spheres_collision.urdf"
arm_model = parser.AddModelsFromUrl(urdf_url)[0]

plant.Finalize()
print(f"模型 '{arm_model}' 的关节数:", plant.num_positions(arm_model))
print(f"模型 '{arm_model}' 的关节数:", plant.num_velocities(arm_model))

# 可视化设置
meshcat = Meshcat()
params = MeshcatVisualizerParams(
    publish_period=0.016,
    role= Role.kIllustration,
    default_color=Rgba(0.9, 0.9, 0.9, 1.0),
    prefix='visualizer',
    delete_on_initialization_event=True
)
MeshcatVisualizer.AddToBuilder(
    builder=builder,
    scene_graph=scene_graph,
    meshcat=meshcat,
    # params=params
)

# 构建系统
diagram = builder.Build()

# 创建模拟器
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)  # 确保实时运行
simulator.get_mutable_integrator().set_maximum_step_size(0.001)  # 减小步长
simulator.get_mutable_integrator().set_fixed_step_mode(True)  # 固定步长

# 设置初始状态
context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(context)

# 设置机械臂初始关节位置（6个关节）
q0 = [0, 0, 1, 1, 0, 0, 0, 1.57, 1, 0, 0, 0, 0, 0]  # 各关节初始角度（弧度）
v0 = [0] * 13 # 各关节初始速度

plant.SetPositions(plant_context, arm_model, q0)
plant.SetVelocities(plant_context, arm_model, v0)

# 运行仿真
try:
    simulator.AdvanceTo(10)  # 模拟5秒
except Exception as e:
    print(f"仿真错误: {e}")

# 打印最终状态
print("最终关节位置:", plant.GetPositions(plant_context, arm_model))
print("最终关节速度:", plant.GetVelocities(plant_context, arm_model))
print("可视化地址:", meshcat.web_url())