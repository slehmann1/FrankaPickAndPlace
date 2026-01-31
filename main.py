import numpy as np
import robosuite as suite

from environments import TargetEnvironment

# --- Initialization ---
env = TargetEnvironment(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    gripper_types="Robotiq85Gripper",
)

obs = env.reset()
env.viewer.set_camera(camera_id=-1)

print("Start")

target_bin_pos = obs.get("target_zone_pos", np.array([0.0, 0.2, 0.82]))

# Where should end effector go?
goal_pos = np.array([0.3, 0.0, 1.0])

# P in PD control
K_P = 4.0

for i in range(1000):
    gripper_pos = obs["robot0_eef_pos"]

    action = np.zeros(7)

    goal_error = goal_pos - gripper_pos
    action[:3] = goal_error * K_P
    action[-1] = -1  # Open

    obs, reward, done, info = env.step(action)
    env.render()
