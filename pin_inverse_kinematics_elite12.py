import time

import numpy as np
import pinocchio
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R

mesh_dir = "/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/elite_description"
urdf_model_path = "/home/gr-arm-7xx2/gr3_arm_ros/catkin_ws/src/elite_description/urdf/xx_elite_12kg.urdf"

# Load the urdf model
model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
    urdf_model_path, mesh_dir
)
print("model name: " + model.name)

# model = pinocchio.buildSampleModelManipulator()
data = model.createData()

frames = model.frames
for i, frame in enumerate(frames):
    print(f'id:{i}, frame:{frame.name}')
JOINT_ID = 6
FRAME_ID = 15
target_R = R.from_quat([0.8897, 0.4560, 0.0183, -0.0027]).as_matrix()
oMdes = pinocchio.SE3(target_R, np.array([-0.6032, 0.3590, 1.0563]))

# q = pinocchio.neutral(model)
q = pinocchio.randomConfiguration(model)
# q = np.array([-3.9, -1.3, 0.6, -0.8, 1.5, -3])
print("start q: " + str(q))

eps = 1e-4
IT_MAX = 1000
DT = 1e-1
damp = 1e-12

i = 0
t = time.time()
while True:
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.framesForwardKinematics(model, data, q)
    iMd = data.oMi[JOINT_ID].actInv(oMdes)
    iMd = data.oMf[FRAME_ID].actInv(oMdes)
    err = pinocchio.log(iMd).vector  # in joint frame
    if norm(err) < eps:
        success = True
        break
    if i >= IT_MAX:
        success = False
        break
    J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  # in joint frame
    J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
    v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
    q = pinocchio.integrate(model, q, v * DT)
    if not i % 10:
        print(f"{i}: error = {err.T}")
    i += 1

if success:
    print("Convergence achieved!")
else:
    print(
        "\n"
        "Warning: the iterative algorithm has not reached convergence "
        "to the desired precision"
    )

print(f"\nresult: {q.flatten().tolist()}")
print(f"\nfinal error: {err.T}")
print(f"总耗时： {time.time() - t}")