import gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder
import cv2

class  VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            render_seq=False,
            render_args={},
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder
        self.render_seq = render_seq
        self.render_args = render_args

        self.step_count = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.frames = list()
        self.step_count = 1
        self.video_recoder.stop()
        return obs
    
    def step(self, action):
        if self.render_seq:
            result = super().step(action['act'])
        else:
            result = super().step(action)
        self.step_count += 1
        if self.file_path is not None \
            and ((self.step_count % self.steps_per_render) == 0):
            if not self.video_recoder.is_ready():
                self.video_recoder.start(self.file_path)

            frame = self.env.render(
                mode=self.mode, **self.render_args).copy()
            if self.render_seq:
                traj_len = action['action_seq'].shape[0]
                xy_action = action['action_seq'].reshape(traj_len, -1, 3)[:, :, :2]
                if "width" in self.render_args:
                    scale_x = self.render_args['width'] / self.env.camera_width
                else:
                    scale_x = 1

                if "height" in self.render_args:
                    scale_y = self.render_args['height'] / self.env.camera_height
                else:
                    scale_y = 1
                for i in range(xy_action.shape[1]):
                    traj = xy_action[:, i, :].copy()
                    traj[:, 0] = traj[:, 0] * scale_x
                    traj[:, 1] = traj[:, 1] * scale_y
                    traj = traj.astype(np.int32)
                    traj = traj.reshape((-1, 1, 2))
                    frame = cv2.polylines(frame, [traj], isClosed=False, color=(255, 0, 0), thickness=1)
                    for j in range(traj.shape[0]):
                        frame = cv2.circle(frame, tuple(traj[j, 0]), 1, (255, 0, 0), -1)
                current_kp = result[0]['agentview_keypoints'].copy().reshape(-1, 3)
                current_kp[:, 0] = current_kp[:, 0] * scale_x
                current_kp[:, 1] = current_kp[:, 1] * scale_y
                current_kp = current_kp.astype(np.int32)

                for i in range(current_kp.shape[0]):
                    frame = cv2.circle(frame, (current_kp[i, 0], current_kp[i, 1]), 1, (0, 0, 255), -1)
                
            if self.mode != "human":
                assert frame.dtype == np.uint8
                self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
