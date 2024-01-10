from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

import robomimic.utils.obs_utils as ObsUtils
from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T

class RobosuiteKpWrapper(EnvRobosuite):
    """Wrapper class for robosuite environments (https://github.com/ARISE-Initiative/robosuite)"""
    def __init__(
        self, 
        env_name,
        render=False, 
        render_offscreen=False, 
        use_image_obs=False, 
        use_depth_obs=False, 
        postprocess_visual_obs=True,
        **kwargs,
    ):
        super().__init__(env_name, render, render_offscreen, use_image_obs, use_depth_obs, postprocess_visual_obs, **kwargs)
        for camera_name in self.env.camera_names:
            self._add_keypoint_observables(camera_name)

    def get_observation(self, di=None):
        """
        Get current environment observation dictionary.

        Args:
            di (dict): current raw observation dictionary from robosuite to wrap and provide 
                as a dictionary. If not provided, will be queried from robosuite.
        """
        if di is None:
            di = self.env._get_observations(force_update=True) if self._is_v1 else self.env._get_observation()
        ret = {}
        for k in di:
            if (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="rgb"):
                # by default images from mujoco are flipped in height
                ret[k] = di[k][::-1]
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)
            elif (k in ObsUtils.OBS_KEYS_TO_MODALITIES) and ObsUtils.key_is_obs_modality(key=k, obs_modality="depth"):
                # by default depth images from mujoco are flipped in height
                ret[k] = di[k][::-1]
                if len(ret[k].shape) == 2:
                    ret[k] = ret[k][..., None] # (H, W, 1)
                assert len(ret[k].shape) == 3 
                # scale entries in depth map to correspond to real distance.
                ret[k] = self.get_real_depth_map(ret[k])
                if self.postprocess_visual_obs:
                    ret[k] = ObsUtils.process_obs(obs=ret[k], obs_key=k)

        # "object" key contains object information
        ret["object"] = np.array(di["object-state"])
        ret["agentview_keypoints"] = di["agentview_keypoints"]

        if self._is_v1:
            for robot in self.env.robots:
                # add all robot-arm-specific observations. Note the (k not in ret) check
                # ensures that we don't accidentally add robot wrist images a second time
                pf = robot.robot_model.naming_prefix
                for k in di:
                    if k.startswith(pf) and (k not in ret) and \
                            (not k.endswith("proprio-state")):
                        ret[k] = np.array(di[k])
        else:
            # minimal proprioception for older versions of robosuite
            ret["proprio"] = np.array(di["robot-state"])
            ret["eef_pos"] = np.array(di["eef_pos"])
            ret["eef_quat"] = np.array(di["eef_quat"])
            ret["gripper_qpos"] = np.array(di["gripper_qpos"])
        return ret

    def _add_keypoint_observables(self, camera_name):

        modality = "keypoint"
        camera_idx = self.env.camera_names.index(camera_name)
        camera_height = self.env.camera_heights[camera_idx]
        camera_width = self.env.camera_widths[camera_idx]
        P = self.get_camera_transform_matrix(camera_name, camera_height, camera_width)
        R = self.get_camera_extrinsic_matrix(camera_name)

        @sensor(modality=modality)
        def keypoints(obs_cache):
            xpos = []
            for geom_name in self.env.nuts[0].visual_geoms:
                xpos.append(self.env.sim.data.get_geom_xpos(geom_name).copy())

            for geom_name in self.env.robots[0].gripper.visual_geoms:
                xpos.append(self.env.sim.data.get_geom_xpos(geom_name).copy())

            xpos = np.array(xpos)
            xpos = np.hstack((xpos, np.ones((xpos.shape[0], 1))))
            projected = P @ xpos.T
            normalized = (projected / projected[2, :]).T
            xpos_camera = (T.pose_inv(R) @ xpos.T).T
            normalized[:, 2] = xpos_camera[:, 2]
            return normalized[:, :3]

        kp_obs = Observable(
            name=f"{camera_name}_keypoints",
            sensor=keypoints,
            sampling_rate=self.env.control_freq,
        )

        self.env.add_observable(kp_obs)
