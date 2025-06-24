# This examples shows how to load and move a robot in meshcat.
# Note: this feature requires Meshcat to be installed, this can be done using
# pip install --user meshcat

import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

# Load the URDF model.
# Conversion with str seems to be necessary when executing this file with ipython
# pinocchio_model_dir = Path(__file__).parent.parent.parent.parent.parent / "Downloads/models"
#
# model_path = pinocchio_model_dir / "example-robot-data/robots"
# mesh_dir = pinocchio_model_dir
# # urdf_filename = "talos_reduced.urdf"
# # urdf_model_path = join(join(model_path,"talos_data/robots"),urdf_filename)
# urdf_filename = "solo.urdf"
# urdf_model_path = model_path / "solo_description/robots" / urdf_filename

mesh_dir = "/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/elite_description"
urdf_model_path = "/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/elite_description/urdf/xx_elite_12kg.urdf"

# Load the urdf model
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir
)
print("model name: " + model.name)

# 初始化 Meshcat 可视化器
viz = MeshcatVisualizer(model, collision_model, visual_model)

# 启动 Meshcat 服务器并打开浏览器窗口
try:
    viz.initViewer(open=True)
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install Python meshcat")
    print(err)
    sys.exit(0)

# 加载机器人模型到可视化器
viz.loadViewerModel()

# 设置初始配置（全零，自由浮动关节在前 7 个位置）
q = pin.neutral(model)
# q = pin.randomConfiguration(model)
q = np.array([-3.9, -1.3, 0.6, -0.8, 1.5, -3])
viz.display(q)
viz.displayVisuals(True)

# 保持可视化窗口打开
# print("Visualization is running. Keep the script alive to view the model.")
# while True:
#     time.sleep(1)

# 运动仿真
q1 = pin.randomConfiguration(model)
v0 = np.random.randn(model.nv) * 2
v0 = np.zeros(model.nv)
data = viz.data
pin.forwardKinematics(model, data, q1, v0)
frame_id = model.getFrameId("tool0")
viz.display()
viz.drawFrameVelocities(frame_id=frame_id)

model.gravity.linear[:] = 0.0
dt = 0.01


def sim_loop():
    tau0 = np.zeros(model.nv)
    tau0 = np.random.randn(model.nv)
    qs = [q1]
    vs = [v0]
    nsteps = 100
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pin.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pin.integrate(model, q, dt * vnext)
        qs.append(qnext)
        vs.append(vnext)
        viz.display(qnext)
        viz.drawFrameVelocities(frame_id=frame_id)
    return qs, vs


qs, vs = sim_loop()


def my_callback(i, *args):
    viz.drawFrameVelocities(frame_id)


# viz.play(qs, dt, callback=my_callback)  # 播放轨迹（不保存视频）

# 保持窗口
# print("按Ctrl+C退出")
# while True:
#     time.sleep(1)

with viz.create_video_ctx("../leap.mp4"):
    viz.play(qs, dt, callback=my_callback)
