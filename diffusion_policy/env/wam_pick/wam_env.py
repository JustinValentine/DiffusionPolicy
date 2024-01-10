import os
import numpy as np
from typing import Optional

import gym
from gym import error, spaces
from gym.utils import seeding
from . import utils
from scipy.spatial.transform import Rotation as R
import mujoco

from diffusion_policy.env.wam_pick.controllers.joint_vel import JointVelocityController


#try:
#    import mujoco_py
#except ImportError as e:
#    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


class WAMEnv(gym.Env):
    def __init__(self, model_path, initial_qlist, frame_skip, robot_mode, n_actions=7, abs_action=False):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}

        self.frame_skip = frame_skip

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()


        self.robot_mode = robot_mode

        
        self.n_joints = 7
        self.n_actions = 7
        self.abs_action = abs_action

        joint_indexes = {
            "joints": list(range(7)),
            "qpos": list(range(7)),
            "qvel": list(range(7)),
        }

        if self.abs_action:
            pass
        else:
            #self.controller = JointVelocityController(self.model, self.data, joint_indexes, actuator_range=(-200, 200), kp=[25, 100, 50, 50, 0.5, 0.000025, 0.0025])
            self.controller = JointVelocityController(self.model, self.data, joint_indexes, actuator_range=(-6.3, 6.3))

        self.distance_threshold = 0.05

        self.initial_qlist = initial_qlist

        self._env_setup()

        self.braking = False

        #self.goal = self._sample_goal()
        self.reset_goal()
        obs = self._get_obs()

        max_abs_action = np.inf
        self.action_space = spaces.Box(-max_abs_action, max_abs_action, shape=(self.n_actions,), dtype='float32')
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        # we are directly manipulating mujoco state here
        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)

        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        for _ in range(self.frame_skip):
            self._set_action(action)
            mujoco.mj_step(self.model, self.data)

        self._step_callback()
        self._render_callback()
        obs = self._get_obs()
        info = self._get_info()

        done = info['is_success']

        reward = self.compute_reward(info['pos'], self.goal, info)
        return obs, reward, done, info

    def reset(self, seed=None):
                
        mujoco.mj_resetData(self.model, self.data)
        self.controller.reset()
        obs = self.reset_model()
        return obs

    def close(self):
        self.render.close()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_id=None, camera_name=None):
        if mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if mode in {
            "rgb_array",
            "depth_array",
        }:

            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None:
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    camera_name,
                )

                self._get_viewer(mode).render(camera_id=camera_id)

        if mode == "rgb_array":
            data = self._get_viewer(mode).read_pixels(depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render()
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                from diffusion_policy.env.wam_pick.rendering import Viewer

                self.viewer = Viewer(self.model, self.data)
            elif mode in {"rgb_array", "depth_array"}:
                from diffusion_policy.env.wam_pick.rendering import RenderContextOffscreen

                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(
                    f"Unexpected mode: {mode}, expected modes: {self.metadata['render_modes']}"
                )

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    # WAM methods
    # ----------------------------

    def _get_joint_angles(self):
        if self.robot_mode == '4dof':
            return utils.get_joint_angles(self.data)[:4]
        else:
            return utils.get_joint_angles(self.data)
    
    def _get_joint_vel(self):
        if self.robot_mode == '4dof':
            return utils.get_joint_vels(self.data)[:4]
        else:
            return utils.get_joint_vels(self.data)


    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.

        if self.reward_type == 'sparse':
            return -1+np.squeeze(info['is_success'])
            
        else:
            d = utils.goal_distance(achieved_goal, goal)
            return -d

    def _set_action(self, action):
        '''
            This method is overridden in action interface subclasses 
        '''
        assert action.shape == (7,)  # not necessarily the same as n_actions, because n_actions < 7 for interface subclasses
        action = action.copy()  # ensure that we don't change the action outside of this scope
        if self.abs_action:
            delta_q = action - self.data.qpos
            k_p = 50
            k_d = 2 * np.sqrt(k_p) 
            lam = np.identity(self.model.nv)
            tau = lam @ (k_p * delta_q - k_d * self.data.qvel)
            self.data.ctrl[:] = tau

            #self.controller.set_goal(delta_q)
            #self.data.ctrl[:] = self.controller.run_controller()
        else:
            torque = self.controller.compute_torque(action)
            self.data.ctrl[:] = torque

            #self.controller.set_goal(action)
            #self.data.ctrl[:] = self.controller.run_controller()
        

    def _get_obs(self):
        """Returns the observation.
        """
        q = utils.get_joint_angles(self.data)
        T = utils.get_kinematics_mat(q, self.n_joints) # end effector frame matrix, computed according to https://support.barrett.com/wiki/WAM/KinematicsJointRangesConversionFactors

        x_pos = T[:3, -1]
        r = R.from_matrix(T[:3, :3])
        orientation = r.as_quat()

        if self.abs_action:
            obs = np.concatenate([x_pos, orientation, self.data.qvel])
        else:
            obs = np.concatenate([x_pos, orientation, q])
            
        return obs.copy()

    def _get_info(self):
        q = utils.get_joint_angles(self.data)
        pos = utils.WAM_fwd_kinematics(q, self.n_joints)
        info = {
            'is_success': self._is_success(pos, self.goal),
            'pos': pos,
            'q_dot': self._get_joint_vel(),
            'q': self._get_joint_angles(),
            'goal': self.goal.flatten(),
        }

        return info

    def _camera_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """



    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position and so forth.
        """

        lookat = self.data.body('wam/wrist_palm_link').xpos

        assert self.viewer is not None
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0
        self.viewer.cam.lookat[:] = lookat



    # Extension methods
    # ----------------------------


    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).

        Implement this in each subclass.
        """
        raise NotImplementedError


    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _reset_constraints(self):
        raise NotImplementedError()

    def _env_setup(self):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass


#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for viewing Physics objects in the DM Control viewer."""

import abc
import enum
import sys
from typing import Dict, Optional

import numpy as np

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768

DEFAULT_WINDOW_TITLE = 'MuJoCo Viewer'

_MAX_RENDERBUFFER_SIZE = 2048


class RenderMode(enum.Enum):
    """Rendering modes for offscreen rendering."""
    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2


class Renderer(abc.ABC):
    """Base interface for rendering simulations."""

    def __init__(self, camera_settings: Optional[Dict] = None):
        self._camera_settings = camera_settings

    @abc.abstractmethod
    def close(self):
        """Cleans up any resources being used by the renderer."""

    @abc.abstractmethod
    def render_to_window(self):
        """Renders the simulation to a window."""

    @abc.abstractmethod
    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """

    def _update_camera(self, camera):
        """Updates the given camera to move to the initial settings."""
        if not self._camera_settings:
            return
        distance = self._camera_settings.get('distance')
        azimuth = self._camera_settings.get('azimuth')
        elevation = self._camera_settings.get('elevation')
        lookat = self._camera_settings.get('lookat')

        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation
        if lookat is not None:
            camera.lookat[:] = lookat



class DMRenderer(Renderer):
    """Class for rendering DM Control Physics objects."""

    def __init__(self, physics, **kwargs):
        super().__init__(**kwargs)
        self._physics = physics
        self._window = None

        # Set the camera to lookat the center of the geoms. (mujoco_py does
        # this automatically.
        if 'lookat' not in self._camera_settings:
            self._camera_settings['lookat'] = [
                np.median(self._physics.data.geom_xpos[:, i]) for i in range(3)
            ]

    def render_to_window(self):
        """Renders the Physics object to a window.

        The window continuously renders the Physics in a separate thread.

        This function is a no-op if the window was already created.
        """
        if not self._window:
            self._window = DMRenderWindow()
            self._window.load_model(self._physics)
            self._update_camera(self._window.camera)
        self._window.run_frame()

    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """
        # TODO(michaelahn): Consider caching the camera.
        camera = mujoco.Camera(
            physics=self._physics,
            height=height,
            width=width,
            camera_id=camera_id)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera(
                camera._render_camera,  # pylint: disable=protected-access
            )

        image = camera.render(
            depth=(mode == RenderMode.DEPTH),
            segmentation=(mode == RenderMode.SEGMENTATION))
        camera._scene.free()  # pylint: disable=protected-access
        return image

    def close(self):
        """Cleans up any resources being used by the renderer."""
        if self._window:
            self._window.close()
            self._window = None


class DMRenderWindow:
    """Class that encapsulates a graphical window."""

    def __init__(self,
                 width: int = DEFAULT_WINDOW_WIDTH,
                 height: int = DEFAULT_WINDOW_HEIGHT,
                 title: str = DEFAULT_WINDOW_TITLE):
        """Creates a graphical render window.

        Args:
            width: The width of the window.
            height: The height of the window.
            title: The title of the window.
        """
        from dm_control import viewer as dmv
        self._viewport = dmv.renderer.Viewport(width, height)
        self._window = dmv.gui.RenderWindow(width, height, title)
        self._viewer = dmv.viewer.Viewer(self._viewport, self._window.mouse,
                                         self._window.keyboard)
        self._draw_surface = None
        self._renderer = dmv.renderer.NullRenderer()

    @property
    def camera(self):
        return self._viewer._camera._camera

    def close(self):
        self._viewer.deinitialize()
        self._renderer.release()
        self._draw_surface.free()
        self._window.close()

    def load_model(self, physics):
        """Loads the given Physics object to render."""
        self._viewer.deinitialize()
        from dm_control import _render
        from dm_control import viewer

        self._draw_surface = _render.Renderer(
            max_width=_MAX_RENDERBUFFER_SIZE, max_height=_MAX_RENDERBUFFER_SIZE)
        self._renderer = viewer.renderer.OffScreenRenderer(
            physics.model, self._draw_surface)

        self._viewer.initialize(physics, self._renderer, touchpad=False)

    def run_frame(self):
        """Renders one frame of the simulation.

        NOTE: This is extremely slow at the moment.
        """
        from dm_control import viewer
        glfw = viewer.gui.glfw_gui.glfw
        glfw_window = self._window._context.window
        if glfw.window_should_close(glfw_window):
            sys.exit(0)

        self._viewport.set_size(*self._window.shape)
        self._viewer.render()
        pixels = self._renderer.pixels

        with self._window._context.make_current() as ctx:
            ctx.call(self._window._update_gui_on_render_thread, glfw_window,
                     pixels)
        self._window._mouse.process_events()
        self._window._keyboard.process_events()
