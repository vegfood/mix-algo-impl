from pydrake.all import (
    DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
    Simulator, Meshcat, MeshcatVisualizer, FindResourceOrThrow
)

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
plant.mutable_gravity_field().set_gravity_vector([0, 0, -9.81])

# 加载URDF
urdf_path = FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf")
with open(urdf_path, 'r') as f:
    print(f.read())  # 确认是否存在 <mass> 和 <inertia> 标签
parser = Parser(plant)
parser.AddModels(urdf_path)
plant.Finalize()

# 可视化
meshcat = Meshcat()
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# 仿真
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)  # 确保实时运行
simulator.get_mutable_integrator().set_maximum_step_size(0.001)  # 减小步长
simulator.get_mutable_integrator().set_fixed_step_mode(True)  # 固定步长

context = simulator.get_mutable_context()
plant_context = plant.GetMyContextFromRoot(context)
plant.SetPositions(plant_context, [180.0])  # 初始角度
plant.SetVelocities(plant_context, [10.0]) # 初始角速度

simulator.AdvanceTo(50.0)
print("当前角度:", plant.GetPositions(plant_context))
print("当前角速度:", plant.GetVelocities(plant_context))
print("Open visualization:", meshcat.web_url())