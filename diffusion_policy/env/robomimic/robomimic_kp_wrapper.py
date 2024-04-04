from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

from scipy.optimize import minimize, lsq_linear, root
import robosuite.utils.transform_utils as T
import mujoco


class RobomimicKPWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        keypoint_obs_key='agentview_keypoints',
        estimate_J=False
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.keypoint_obs_key = keypoint_obs_key 
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.estimate_J = estimate_J

        # setup spaces
        max_camera_dim = max(max(env.env.camera_widths), max(env.env.camera_heights))
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=0,
            high=max_camera_dim,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space
        self.prev_action =  np.zeros(action_space.shape)

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith("keypoints"):
                min_value, max_value = 0, max_camera_dim
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space
        self.J_initialized = False
        self.keypoint_cache = None
        self.alpha = 0.25
        self.P = np.eye(8)
        self.j_weight = 0.8
        
        self.gripper_controller = GripperController(self.env.env.sim, self.env.env.robots[0])
        for robot in self.env.env.robots:
            if robot.has_gripper:
                robot.grip_action = self.gripper_controller.grip_action # dynamically overload grip action
        camera_name = 'agentview'
        camera_idx = self.env.env.camera_names.index(camera_name)
        self.camera_height = self.env.env.camera_heights[camera_idx]
        self.camera_width = self.env.env.camera_widths[camera_idx]



    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        raw_obs[self.keypoint_obs_key] = raw_obs[self.keypoint_obs_key].reshape(-1)
        
        self.render_cache = raw_obs[self.render_obs_key]
        self.keypoint_cache = raw_obs[self.keypoint_obs_key][-9:].copy()
        #self.keypoint_cache = raw_obs[self.keypoint_obs_key].copy()
        #self.keypoint_cache[2::3] *= 84

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs
    

    def get_robot_J(self, eef_name='gipper0_site', idx=8):
        # problem, will only work for geoms, not sites

        J_pos = np.array(self.env.env.sim.data.get_geom_jacp(eef_name).reshape((3, -1))[:, :idx])
        J_ori = np.array(self.env.env.sim.data.get_geom_jacr(eef_name).reshape((3, -1))[:, :idx])
        J_full = np.array(np.vstack([J_pos, J_ori]))

        return J_full

    
    def get_true_image_jacobian(self):

        camera_name = 'agentview'
        camera_idx = self.env.env.camera_names.index(camera_name)
        camera_height = self.env.env.camera_heights[camera_idx]
        camera_width = self.env.env.camera_widths[camera_idx]
        P = self.env.get_camera_transform_matrix(camera_name, camera_height, camera_width)
        R = self.env.get_camera_extrinsic_matrix(camera_name)
        K = self.env.get_camera_intrinsic_matrix(camera_name, camera_height, camera_width)

        K_mod = np.eye(3)
        K_mod[:2, :2] = K[:2, :2]

        external = T.pose_inv(R)
        rot = external[:3, :3]
        trans = R[:3, -1]
        skew_matrix = np.array([[0, -trans[2], trans[1]],
                            [trans[2], 0, -trans[0]],
                            [-trans[1], trans[0], 0]])

        V = np.hstack([rot, np.zeros_like(rot)]) # ignore angular

        xpos = []
        J_list = []

        xpos = []
        for entity_type, name in self.env.kp_entities:
            if entity_type == mujoco.mjtObj.mjOBJ_GEOM:
                xpos.append(self.env.env.sim.data.get_geom_xpos(name).copy())
            elif entity_type == mujoco.mjtObj.mjOBJ_SITE:
                xpos.append(self.env.env.sim.data.get_site_xpos(name).copy())
            else:
                raise NotImplementedError(f"type {entity_type} not implemented")
            J = self.get_robot_J(name, 9)
            if not (J[:, 8] == 0).all(): # account for right finger
                J = J[:, [0, 1, 2, 3, 4, 5, 6, 8]]
                J[:, -1] *= -1
                J_list.append(J)
            else:
                J_list.append(J[:, [0, 1, 2, 3, 4, 5, 6, 7]])

        L = []
        for i, coords in enumerate(xpos):

            coords = np.concatenate((coords, [1])).reshape(-1, 1)

            xpos_camera = T.pose_inv(R) @ coords
            X, Y, Z = xpos_camera[:3, 0]
            L_x = np.array([
                [1/Z, 0, -X/(Z*Z)], 
                [0, 1/Z, -Y/(Z*Z)],
                [0, 0, 1]
            ])

            L.append(K_mod @ L_x @ V @ J_list[i])
        L = np.vstack(L)

        return L


    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        self.J_initialized = False
        return obs

    def control(self, desired_qpos):

        epsilon = 0.001
        max_iter = 1
        min_iter = 1

        #n_iter = 0
        #while (n_iter < min_iter) or (np.linalg.norm(desired_qpos - self.get_qpos()) > epsilon) and (n_iter < max_iter):
        #    delta = desired_qpos - self.get_qpos()
        #    delta[7] = -delta[7]
        #    raw_obs, reward, done, info = self.env.step(delta) # relative to current position
        #    n_iter += 1
        velocity = desired_qpos.copy()
        velocity[:7] = velocity[:7] / self.env.env.control_timestep

        self.gripper_controller.policy_step = True
        raw_obs, reward, done, info = self.env.step(velocity) # relative to current position
        return raw_obs, reward, done, info
    
    def step(self, action):

        if self.estimate_J and not self.J_initialized:
            self.initialize_J()

        joint_ranges = np.array([
        [-2.8973,  2.8973],
       [-1.7628,  1.7628],
       [-2.8973,  2.8973],
       [-3.0718, -0.0698],
       [-2.8973,  2.8973],
       [-0.0175,  3.7525],
       [-2.8973,  2.8973]]
        )

        action_scaled = action[-9:].copy()
        #action_scaled = action.copy()

        # bring desired fingers closer
        finger2 = action_scaled[-3:]
        finger1 = action_scaled[-6:-3]
        grip_displacement = finger2 - finger1
        action_scaled[-3:] = finger2 - 0.05*grip_displacement
        action_scaled[-6:-3] = finger1 + 0.05*grip_displacement
        #action_scaled[2::3] *= 84
        
        
        bounds_l = joint_ranges[:, 0]
        bounds_r = joint_ranges[:, 1]

        j_centers = (bounds_l + bounds_r) / 2
        j_range = bounds_r - bounds_l

        lam = 0.1
        W_sing = 0.1 * np.eye(8)
        #W_sing[7, 7] *= 100 # gripper joint strong weight since much smaller range


        W_limits = np.ones(7) * 0.1
        d_lower = 0.25
        d_upper = 0.75
        n_iter = 2


        for i in range(n_iter):

            error = self.keypoint_cache - action_scaled

            if not self.estimate_J:
                self.J = self.get_true_image_jacobian()[-9:, :]
            
            
            #dq, residuals, rank, s = np.linalg.lstsq(self.J, error, rcond=None)
            dq, residuals, rank, s = np.linalg.lstsq(self.J.T @ self.J + W_sing, self.J.T @ error, rcond=None)
            dq = -0.5*dq

            qpos = self.get_qpos()

            v = W_limits / j_range
            greater_mask = qpos[:7] >= j_centers
            v[greater_mask] *= -1


            #null_p = (np.eye(7) - J_pinv @ J_full)
            #null_q = null_p @ v
            #dq[:7] = dq[:7] - (np.eye(7) - np.linalg.pinv(J_full) @ J_full) @ dq[:7]
            #dq[:7] = dq[:7] + null_q

            desired_qpos = qpos + dq

            prev_position = np.array(qpos)

            raw_obs, reward, done, info = self.control(dq)

            obs = self.get_observation()
            qpos = self.get_qpos()
            dx = np.array(qpos - prev_position).reshape(-1, 1)

            new_error = self.keypoint_cache - action_scaled
            dy = (new_error - error).reshape(-1, 1)

            if self.estimate_J:
                update = ((dy - self.J @ dx) @ dx.T @ self.P) / (self.j_weight + dx.T @ self.P @ dx)
                self.P = (1 / self.j_weight) * (self.P - (self.P @ dx @ dx.T @ self.P) / (self.j_weight + dx.T @ self.P @ dx)) 
                self.J += 1.0*update

        return obs, reward, done, info

    def render(self, mode='rgb_array', height=84, width=84):
        img = self.env.render(mode=mode, height=height, width=width)
        return img

    
    def get_qpos(self):
        robot_qpos = self.env.env.robots[0]._joint_positions.copy()
        gripper_indexes = self.env.env.robots[0]._ref_gripper_joint_pos_indexes.copy()
        gripper_qpos = self.env.env.sim.data.qpos[gripper_indexes].copy()
        return np.append(robot_qpos, gripper_qpos[0])
    
    def initialize_J(self):
        epsilon = 0.1 # consider getting from action range
        #self.J = np.zeros((self.action_space.shape[0], self.env.action_dimension))
        self.J = np.zeros((9, self.env.action_dimension))
        self.c_diffJ = np.zeros((self.action_space.shape[0], self.env.action_dimension))

        state = self.env.get_state()['states']
        for i in range(self.env.action_dimension):
            e_i = np.zeros(self.env.action_dimension)
            e_i[i] = epsilon
            desired_qpos = self.get_qpos() + e_i
            n_iter = 0
            if i == (self.env.action_dimension - 1):
                robot = self.env.env.robots[0]
                grip_idx = robot._ref_gripper_joint_pos_indexes
                self.env.env.sim.data.qpos[grip_idx] = np.array([0.03, -0.03])
                self.env.env.sim.forward()
            else:
                self.env.env.robots[0].set_robot_joint_positions(desired_qpos[:7])
            #while ((abs(self.get_qpos()[i] - desired_qpos[i]) > epsilon*.1) and (n_iter < 20)):
            #    delta = desired_qpos - self.get_qpos()
            #    raw_obs, reward, done, info = self.env.step(delta) # relative to current position
            #    n_iter += 1
            self.get_observation()
            x_n = self.get_qpos()
            f_n = self.keypoint_cache.copy()

            self.env.reset_to({'states': state})

            desired_qpos = self.get_qpos() - e_i
            n_iter = 0
            if i == (self.env.action_dimension - 1):
                robot = self.env.env.robots[0]
                grip_idx = robot._ref_gripper_joint_pos_indexes
                self.env.env.sim.data.qpos[grip_idx] = np.array([0.01, -0.01])
                self.env.env.sim.forward()
            else:
                self.env.env.robots[0].set_robot_joint_positions(desired_qpos[:7])
            #while ((abs(self.get_qpos()[i] - desired_qpos[i]) > epsilon*.1) and (n_iter < 20)):
            #    delta = desired_qpos - self.get_qpos()
            #    raw_obs, reward, done, info = self.env.step(delta) # relative to current position
            #    n_iter += 1

            self.get_observation()
            x_n_1 = self.get_qpos()
            f_n_1 = self.keypoint_cache.copy()
            dx = x_n[i] - x_n_1[i]
            self.J[:, i] = (f_n - f_n_1) / (dx)

            #dy = (f_n - f_n_1).reshape(-1, 1)
            #dx = (x_n - x_n_1).reshape(-1, 1)

            #update = ((dy - self.J @ dx) @ dx.T @ self.P) / (0 + dx.T @ self.P @ dx)
            #self.P = (1 / 0.5) * (self.P - (self.P @ dx @ dx.T @ self.P) / (0.5 + dx.T @ self.P @ dx)) 
            #self.J += 1.0*update

            self.env.reset_to({'states': state})

        self.J_initialized = True

class GripperController:
    def __init__(self, sim, robot, kp=100, dampening=0.01):
        self.sim = sim
        self.robot = robot
        self.policy_step = True
        self.kp = kp
        self.kd = 2 * np.sqrt(kp) * dampening
        
    def grip_action(self, gripper, gripper_action):
        """
        Executes @gripper_action for specified @gripper

        Args:
            gripper (GripperModel): Gripper to execute action for
            gripper_action (float): Value between [-1,1] to send to gripper
        """

        grip_pos_idx = self.robot._ref_gripper_joint_pos_indexes
        grip_vel_idx = self.robot._ref_gripper_joint_vel_indexes
        qpos = self.sim.data.qpos[grip_pos_idx].copy()
        qvel = self.sim.data.qvel[grip_vel_idx].copy()

        if self.policy_step:
            self.desired_qpos = qpos + (np.array([1, -1]) * gripper_action)

        actuator_idxs = [self.sim.model.actuator_name2id(actuator) for actuator in gripper.actuators]


        #gripper_action_actual = gripper.format_action(gripper_action)
        # rescale normalized gripper action to control ranges
        #ctrl_range = self.sim.model.actuator_ctrlrange[actuator_idxs]
        #bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        #weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        #applied_gripper_action = bias + weight * gripper_action_actual
        ctrl_range = self.sim.model.actuator_ctrlrange[actuator_idxs]

        position_error = self.desired_qpos - qpos
        vel_pos_error = -qvel

        gripper_action_raw = self.kp * position_error + self.kd * vel_pos_error
        applied_gripper_action = np.clip(gripper_action_raw, ctrl_range[:, 0], ctrl_range[:, 1])
        self.sim.data.ctrl[actuator_idxs] = applied_gripper_action
        #self.env.env.sim.data.ctrl[actuator_idxs] = [0.03, -0.03]

        self.policy_step = False

def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
