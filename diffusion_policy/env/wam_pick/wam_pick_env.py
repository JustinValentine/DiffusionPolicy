

import os
from gym import utils as gym_utils
import numpy as np
from diffusion_policy.env.wam_pick.wam_env import WAMEnv
from . import utils
from .utils import WAM_fwd_kinematics
import mujoco



MODEL_XML_PATH = os.path.join('wam', 'wam_7dof_wam_bhand_frictionless.xml')

class WAMPickEnv(WAMEnv, gym_utils.EzPickle):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, reward_type='sparse', model_path=MODEL_XML_PATH, robot_mode='7dof', frame_skip=10, n_actions=7, abs_action=False
    ):
        """Initializes a new WAM environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        
        self.obj_range = None
        self.target_range = None
        self.reward_type = reward_type
        self.goal_x_min, self.goal_x_max = (0.4289, 0.8001)
        self.goal_y_min, self.goal_y_max = (-0.2741, 0.1973)
        self.goal_z = 0.100

        initial_qlist = [ 0.00522284, -0.14623045, -0.03904396,  1.67928286,  0.06278251,
            -0.0619918 ,  0.03524149]
        super(WAMPickEnv, self).__init__(
            model_path=model_path, initial_qlist=initial_qlist, frame_skip=frame_skip, robot_mode=robot_mode, n_actions=n_actions, abs_action=abs_action)

        gym_utils.EzPickle.__init__(self)

            

    # WAMEnv methods
    # ----------------------------
    def _check_constraints(self):
        return

    def _step_callback(self):
        self._check_constraints()
        return


    def _render_callback(self):
        # Visualize target.
        self.model.site('target0').pos = self.goal + np.array([0, 0, 0.346*3]) # make note of what each of these numbers represent

        # Add marker to end-effector
        q = utils.get_joint_angles(self.data)
        ee_pos = WAM_fwd_kinematics(q, self.n_joints)
        self.model.site('ee').pos = ee_pos + np.array([0, 0, 0.346*3])

        mujoco.mj_forward(self.model, self.data)
    

    def reset_goal(self):
        g = self._sample_goal()
        self.goal = g


    def set_goal(self, goal):
        self.goal = goal

    def _sample_goal(self):
        goal = np.array([
            self.np_random.uniform(self.goal_x_min, self.goal_x_max),
            self.np_random.uniform(self.goal_y_min, self.goal_y_max),
            self.goal_z
        ])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = utils.goal_distance(achieved_goal, desired_goal)
        success = True # innocent until proven guilty
        if d >= self.distance_threshold:
            success = False
        
        return success


    def reset_model(self):
        initial_qpos = self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq) + self.initial_qlist 
        initial_qvel = np.zeros(self.model.nq)
        self.set_state(initial_qpos, initial_qvel)
        self.reset_goal()

        return self._get_obs()


    def _env_setup(self):
        initial_qpos = self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq) + self.initial_qlist 

        self.data.qpos[:] = initial_qpos
            
        mujoco.mj_forward(self.model, self.data)


    def render(self, mode='human', width=500, height=500):
        return super(WAMPickEnv, self).render(mode, width, height)
