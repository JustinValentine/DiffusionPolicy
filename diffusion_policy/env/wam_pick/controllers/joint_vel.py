from collections.abc import Iterable

import numpy as np

from diffusion_policy.env.wam_pick.utils import RingBuffer


class JointVelocityController:
    """
    Controller for controlling the robot arm's joint velocities. This is simply a P controller with desired torques
    (pre gravity compensation) taken to be proportional to the velocity error of the robot joints.

    NOTE: Control input actions assumed to be taken as absolute joint velocities. A given action to this
    controller is assumed to be of the form: (vel_j0, vel_j1, ... , vel_jn-1) for an n-joint robot

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or list of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or list of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or list of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or list of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or list of float): velocity gain for determining desired torques based upon the joint vel errors.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        velocity_limits (2-list of float or 2-list of list of floats): Limits (m/s) below and above which the magnitude
            of a calculated goal joint velocity will be clipped. Can be either be a 2-list (same min/max value for all
            joint dims), or a 2-list of list (specific min/max values for each dim)

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error
    """

    def __init__(
        self,
        model,
        data,
        joint_indexes,
        actuator_range,
        kp=0.25,
        velocity_limits=None,
        **kwargs,  # does nothing; used so no error raised when dict is passed with extra terms used previously
    ):
        self.model = model
        self.data = data
        # Control dimension
        self.control_dim = len(joint_indexes["joints"])
        self.joint_dim = len(joint_indexes["joints"])

        self.actuator_range = self.model.actuator_ctrlrange
        self.actuator_min = actuator_range[0]
        self.actuator_max = actuator_range[1]

        self.joint_index = joint_indexes["joints"]
        self.qpos_index = joint_indexes["qpos"]
        self.qvel_index = joint_indexes["qvel"]

        # gains and corresopnding vars
        self.kp = self.nums2array(kp, self.joint_dim)
        # if kp is a single value, map wrist gains accordingly (scale down x10 for final two joints)

        if isinstance(kp, float) or isinstance(kp, int):
            # Scale kpp according to how wide the actuator range is for this robot
            self.kp = kp * (self.actuator_range[:, 1] - self.actuator_range[:, 0])
        self.ki = self.kp * 0.005
        self.kd = self.kp * 0.001
        self.last_err = np.zeros(self.joint_dim)
        self.derr_buf = RingBuffer(dim=self.joint_dim, length=5)
        self.summed_err = np.zeros(self.joint_dim)
        self.saturated = False

        # limits
        self.velocity_limits = (
            np.array(velocity_limits) if velocity_limits is not None else None
        )

    def compute_torque(self, goal_vel):
        """
        Calculates the torques required to reach the desired setpoint

        Returns:
             np.array: Command torques
        """

        if self.velocity_limits is not None:
            goal_vel = np.clip(
                self.goal_vel, self.velocity_limits[0], self.velocity_limits[1]
            )

        # Update state
        joint_vel = np.array(self.data.qvel[self.qvel_index])

        # Compute necessary error terms for PID velocity controller
        err = goal_vel - joint_vel
        derr = err - self.last_err
        self.last_err = err
        self.derr_buf.push(derr)

        # Only add to I component if we're not saturated (anti-windup)
        if not self.saturated:
            self.summed_err += err

        grav_compensation = self.data.qfrc_bias[self.qvel_index]
        # Compute command torques via PID velocity controller plus gravity compensation torques
        torques_raw = (
            self.kp * err
            + self.ki * self.summed_err
            + self.kd * self.derr_buf.average
            + grav_compensation
        )

        # Clip torques
        torques = np.clip(torques_raw, self.actuator_range[:, 0], self.actuator_range[:, 1])

        # Check if we're saturated
        self.saturated = False if np.sum(np.abs(torques - torques_raw)) == 0 else True

        # Return final torques
        return torques

    def reset(self):
        self.derr_buf.clear()
        self.last_err = np.zeros(self.joint_dim)
        self.summed_err = np.zeros(self.joint_dim)
        self.saturated = False

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError(
                "Error: Only numeric inputs are supported for this function, nums2array!"
            )

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums


# import abc
# from collections.abc import Iterable

# import mujoco
# import numpy as np

# import robosuite.macros as macros


# class Controller(object, metaclass=abc.ABCMeta):
#    """
#    General controller interface.

#    Requires reference to mujoco sim object, eef_name of specific robot, relevant joint_indexes to that robot, and
#    whether an initial_joint is used for nullspace torques or not

#    Args:
#        sim (MjSim): Simulator instance this controller will pull robot state updates from

#        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

#        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

#            :`'joints'`: list of indexes to relevant robot joints
#            :`'qpos'`: list of indexes to relevant robot joint positions
#            :`'qvel'`: list of indexes to relevant robot joint velocities

#        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
#    """

#    def __init__(
#        self,
#        sim,
#        eef_name,
#        joint_indexes,
#        actuator_range,
#    ):

#        # Actuator range


#        # Attributes for scaling / clipping inputs to outputs
#        self.action_scale = None
#        self.action_input_transform = None
#        self.action_output_transform = None

#        # Private property attributes
#        self.control_dim = None
#        self.output_min = None
#        self.output_max = None
#        self.input_min = None
#        self.input_max = None

#        # mujoco simulator state
#        self.sim = sim
#        self.model_timestep = macros.SIMULATION_TIMESTEP
#        self.eef_name = eef_name
#        self.joint_index = joint_indexes["joints"]
#        self.qpos_index = joint_indexes["qpos"]
#        self.qvel_index = joint_indexes["qvel"]

#        # robot states
#        self.ee_pos = None
#        self.ee_ori_mat = None
#        self.ee_pos_vel = None
#        self.ee_ori_vel = None
#        self.joint_pos = None

#        # dynamics and kinematics
#        self.J_pos = None
#        self.J_ori = None
#        self.J_full = None
#        self.mass_matrix = None

#        # Joint dimension
#        self.joint_dim = len(joint_indexes["joints"])

#        # Torques being outputted by the controller
#        self.torques = None

#        # Update flag to prevent redundant update calls
#        self.new_update = True

#        # Initialize controller by updating internal state and setting the initial joint, pos, and ori
#        self.update()
#        self.initial_joint = self.joint_pos
#        self.initial_ee_pos = self.ee_pos
#        self.initial_ee_ori_mat = self.ee_ori_mat

#    @abc.abstractmethod
#    def run_controller(self):
#        """
#        Abstract method that should be implemented in all subclass controllers, and should convert a given action
#        into torques (pre gravity compensation) to be executed on the robot.
#        Additionally, resets the self.new_update flag so that the next self.update call will occur
#        """
#        self.new_update = True

#    def scale_action(self, action):
#        """
#        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
#        the range self.output_min and self.output_max

#        Args:
#            action (Iterable): Actions to scale

#        Returns:
#            np.array: Re-scaled action
#        """

#        if self.action_scale is None:
#            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
#            self.action_output_transform = (self.output_max + self.output_min) / 2.0
#            self.action_input_transform = (self.input_max + self.input_min) / 2.0
#        action = np.clip(action, self.input_min, self.input_max)
#        transformed_action = (action - self.action_input_transform) * self.action_scale + self.action_output_transform

#        return transformed_action

#    def update(self, force=False):
#        """
#        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
#        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
#        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
#        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
#        self.new_update flag

#        Args:
#            force (bool): Whether to force an update to occur or not
#        """

#        # Only run update if self.new_update or force flag is set
#        if self.new_update or force:
#            self.sim.forward()

#            self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])
#            self.ee_ori_mat = np.array(
#                self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_name)].reshape([3, 3])
#            )
#            self.ee_pos_vel = np.array(self.sim.data.get_site_xvelp(self.eef_name))
#            self.ee_ori_vel = np.array(self.sim.data.get_site_xvelr(self.eef_name))

#            self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
#            self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

#            self.J_pos = np.array(self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index])
#            self.J_ori = np.array(self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index])
#            self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

#            mass_matrix = np.ndarray(shape=(self.sim.model.nv, self.sim.model.nv), dtype=np.float64, order="C")
#            mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
#            mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
#            self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]

#            # Clear self.new_update
#            self.new_update = False

#    def update_base_pose(self, base_pos, base_ori):
#        """
#        Optional function to implement in subclass controllers that will take in @base_pos and @base_ori and update
#        internal configuration to account for changes in the respective states. Useful for controllers e.g. IK, which
#        is based on pybullet and requires knowledge of simulator state deviations between pybullet and mujoco

#        Args:
#            base_pos (3-tuple): x,y,z position of robot base in mujoco world coordinates
#            base_ori (4-tuple): x,y,z,w orientation or robot base in mujoco world coordinates
#        """
#        pass

#    def update_initial_joints(self, initial_joints):
#        """
#        Updates the internal attribute self.initial_joints. This is useful for updating changes in controller-specific
#        behavior, such as with OSC where self.initial_joints is used for determine nullspace actions

#        This function can also be extended by subclassed controllers for additional controller-specific updates

#        Args:
#            initial_joints (Iterable): Array of joint position values to update the initial joints
#        """
#        self.initial_joint = np.array(initial_joints)
#        self.update(force=True)
#        self.initial_ee_pos = self.ee_pos
#        self.initial_ee_ori_mat = self.ee_ori_mat

#    def clip_torques(self, torques):
#        """
#        Clips the torques to be within the actuator limits

#        Args:
#            torques (Iterable): Torques to clip

#        Returns:
#            np.array: Clipped torques
#        """
#        return

#    def reset_goal(self):
#        """
#        Resets the goal -- usually by setting to the goal to all zeros, but in some cases may be different (e.g.: OSC)
#        """
#        raise NotImplementedError


#    @property
#    def torque_compensation(self):
#        """
#        Gravity compensation for this robot arm

#        Returns:
#            np.array: torques
#        """
#        return self.sim.data.qfrc_bias[self.qvel_index]

#    @property
#    def actuator_limits(self):
#        """
#        Torque limits for this controller

#        Returns:
#            2-tuple:

#                - (np.array) minimum actuator torques
#                - (np.array) maximum actuator torques
#        """
#        return self.actuator_min, self.actuator_max

#    @property
#    def control_limits(self):
#        """
#        Limits over this controller's action space, which defaults to input min/max

#        Returns:
#            2-tuple:

#                - (np.array) minimum action values
#                - (np.array) maximum action values
#        """
#        return self.input_min, self.input_max

#    @property
#    def name(self):
#        """
#        Name of this controller

#        Returns:
#            str: controller name
#        """
#        raise NotImplementedError
