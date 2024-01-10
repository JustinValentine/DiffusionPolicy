from typing import List, Dict, Optional, Optional
import numpy as np
import gym
from gym.spaces import Box

class WAMLowdimWrapper(gym.Env):
    def __init__(self,
            env,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            init_pick_loc: Optional[np.ndarray]=None,
            render_hw = (240,360)
        ):
        self.env = env
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.init_pick_loc = init_pick_loc
        self.render_hw = render_hw

        self.pix_min_w, self.pix_max_w = (0.07091563, 0.85618893) # table boundaries in transformed pix space
        self.pix_min_h, self.pix_max_h = (-0.10631764, 0.92036586)

        obs_shape = (env.observation_space.shape[0] + 2,)
        self.obs_space = Box(-np.inf, np.inf, shape=obs_shape, dtype='float32')

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.obs_space

    def seed(self, seed=None):
        return self.env.seed(seed)


    def _update_obs(self, obs):

        # goal to pix mapping
        goal = self.env.goal.copy()
        goal_norm_h = ((goal[0] - self.env.goal_x_min) / (self.env.goal_x_max - self.env.goal_x_min))
        goal_norm_w = ((goal[1] - self.env.goal_y_min) / (self.env.goal_y_max - self.env.goal_y_min))

        goal_pix_h = self.pix_min_h + goal_norm_h * (self.pix_max_h - self.pix_min_h)
        goal_pix_w = self.pix_min_w + goal_norm_w * (self.pix_max_w - self.pix_min_w)

        obs = np.concatenate([obs, np.array([goal_pix_h, goal_pix_w])])
        return obs


    def _pix_to_goal(self, pick_loc):
        goal_norm_h = ((pick_loc[0] - self.pix_min_h) / (self.pix_max_h - self.pix_min_h))
        goal_norm_w = ((pick_loc[1] - self.pix_min_w) / (self.pix_max_w - self.pix_min_w))

        goal_x = self.env.goal_x_min + goal_norm_h * (self.env.goal_x_max - self.env.goal_x_min)
        goal_y = self.env.goal_y_min + goal_norm_w * (self.env.goal_y_max - self.env.goal_y_min)
        return np.array([goal_x, goal_y, self.env.goal_z])

    def reset(self):
        if self.init_qpos is not None:
            # reset anyway to be safe, not very expensive
            _ = self.env.reset()
            # start from known state
            goal = self._pix_to_goal(self.init_pick_loc)
            self.env.set_goal(goal)
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env._get_obs()
            obs = self._update_obs(obs)
            return obs
        else:
            obs = self.env.reset()
            obs = self._update_obs(obs)
            return obs

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, width=w, height=h)
    
    def step(self, a):
        obs, reward, done, info = self.env.step(a)
        obs = self._update_obs(obs)
        return obs, reward, done, info
