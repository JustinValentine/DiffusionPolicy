import gym
import numpy as np
from diffusion_policy.real_world.video_recorder import VideoRecorder
import cv2

class VideoRecordingWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            video_recoder: VideoRecorder,
            mode='rgb_array',
            file_path=None,
            steps_per_render=1,
            render_seq=True,
            **kwargs
        ):
        """
        When file_path is None, don't record.
        """
        super().__init__(env)
        
        self.mode = mode
        self.render_kwargs = kwargs
        self.steps_per_render = steps_per_render
        self.file_path = file_path
        self.video_recoder = video_recoder
        self.render_seq = render_seq

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
                mode=self.mode, **self.render_kwargs)
            if self.render_seq:
                traj_len = action['action_seq'].shape[0]
                xy_action = action['action_seq'].reshape(traj_len, -1, 3)[:, :, :2]
                #scale = self.env.render_size / self.env.window_size
                scale = 1
                for i in range(xy_action.shape[1]):
                    traj = xy_action[:, i, :]
                    traj = (traj * scale).astype(np.int32)
                    traj = traj.reshape((-1, 1, 2))
                    frame = cv2.polylines(frame, [traj], isClosed=False, color=(255, 0, 0), thickness=1)
            if self.mode != "human":
                assert frame.dtype == np.uint8
                self.video_recoder.write_frame(frame)
        return result
    
    def render(self, mode='rgb_array', **kwargs):
        if self.video_recoder.is_ready():
            self.video_recoder.stop()
        return self.file_path
