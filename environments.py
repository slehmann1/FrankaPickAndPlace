import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask


class TargetEnvironment(ManipulationEnv):
    """An Environment containing a green target area for an object to be placed in.

    Args:
        ManipulationEnv (RobotEnv): Robosuite Manipulation Environment
    """

    TABLE_HEIGHT = 0.8

    def _load_model(self):
        super()._load_model()

        # Create environment
        self.mujoco_arena = TableArena(
            table_full_size=(0.8, 0.8, 0.05), table_offset=(0, 0, self.TABLE_HEIGHT)
        )
        self.target_zone = BoxObject(
            name="target_zone", size=[0.15, 0.15, 0.05], rgba=[0, 1, 0, 1], joints=None
        )

        # Reposition for environment offset
        self.robots[0].robot_model.set_base_xpos([-0.5, 0, 0])

        robot_models = [robot.robot_model for robot in self.robots]
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=robot_models,
            mujoco_objects=[self.target_zone],
        )

    def _setup_references(self):
        super()._setup_references()
        # Grab the physics ID for the target zone
        self.target_zone_body_id = self.sim.model.body_name2id("target_zone_main")

    def _reset_internal(self):
        super()._reset_internal()
        # Place target zone on table
        self.sim.model.body_pos[self.target_zone_body_id] = np.array(
            [0.2, 0.0, self.TABLE_HEIGHT + 0.02]
        )

    # Abstract methods that need to be implemented
    def reward(self, action=None):
        """
        Required by robosuite. Returning 0 for now.
        """
        return 0.0

    def _check_success(self):
        """
        Required by robosuite. (Is the task done) Returning False for now.
        """
        return False
