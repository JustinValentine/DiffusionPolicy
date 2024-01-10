
from typing import Dict, Optional
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np
import cv2

class PushTImageKpActionEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None
            ):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        Dblockkps = np.prod(local_keypoint_map['block'].shape) + 2 # include agent pos

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'low_dim': spaces.Box(
                low=0,
                high=ws,
                shape=(Dblockkps,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Box(
            low=0,
            high=ws,
            shape=(Dblockkps,),
            dtype=np.float32
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None
        self.render_cache = None
        self.keypoint_cache = None

        self.J_initialized = False


    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs
    
    def _get_obs(self):

        # get keypoints
        obj_map = {
            'block': self.block
        }

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        self.draw_kp_map = draw_kp_map
        
        agent_pos = np.array(self.agent.position)
        # construct keypoint obs
        keypoint_obs = np.concatenate([kps.flatten(), agent_pos], axis=-1)

        img = super()._render_frame(mode='rgb_array')

        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            "low_dim": keypoint_obs
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img
        self.keypoint_cache = keypoint_obs

        return obs
    
    def step(self, action):

        if not self.J_initialized:
            self.initialize_J()

        for i in range(5):

            error = action - self.keypoint_cache
            dq, residuals, rank, s = np.linalg.lstsq(self.J, error, rcond=None)

            robot_action = self.agent.position - dq

            robot_action = np.clip(robot_action, 0, self.window_size)
            prev_position = np.array(self.agent.position)
            observation, reward, done, info = super().step(robot_action)
            dx = np.array(self.agent.position - prev_position).reshape(-1, 1)
            new_error = action - self.keypoint_cache
            dy = (new_error - error).reshape(-1, 1)
            #dq = dq.reshape(-1, 1)
            update = ((dy - self.J @ dx) @ dx.T) / (dx.T @ dx)
            self.J += update

        return observation, reward, done, info

    def initialize_J(self):
        epsilon = 10
        current_pos = np.array(self.agent.position)
        self.J = np.zeros((20, 2))
        for i in range(current_pos.shape[0]):
            e_i = np.zeros(2)
            e_i[i] = epsilon
            new_pos = current_pos + e_i
            super().step(new_pos)
            f_n = self.keypoint_cache.copy()
            e_i = np.zeros(2)
            e_i[i] = epsilon
            new_pos = current_pos - e_i
            super().step(new_pos)
            f_n_1 = self.keypoint_cache.copy()
            self.J[:, i] = (f_n - f_n_1) / (2 * epsilon)
        self.J_initialized = True

    def reset(self):

        self.J_initialized = False

        return super().reset()

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()

        img = self.render_cache.copy()
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/self.render_size))
        
        return img
