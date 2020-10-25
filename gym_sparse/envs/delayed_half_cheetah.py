import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class DelayedHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, steps_threshold=10):
        self.steps_threshold = steps_threshold
        self.steps_taken = 0
        self.accumulated_reward = 0

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        # Move the model
        self.steps_taken += 1
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()

        # Compute the control and run rewards
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False

        # Give the reward only every Nth step
        if self.steps_taken % self.steps_threshold == 0:
            reward += self.accumulated_reward
            self.accumulated_reward = 0
        else:
            self.accumulated_reward += reward
            reward = 0

        return ob, reward, done, dict(reward_run=reward_run,
                                      reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1,
                                                       high=.1,
                                                       size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        self.steps_taken = 0
        self.accumulated_reward = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
