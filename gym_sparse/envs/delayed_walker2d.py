import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class DelayedWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, steps_threshold=10):
        self.steps_threshold = steps_threshold
        self.steps_taken = 0
        self.accumulated_reward = 0

        mujoco_env.MujocoEnv.__init__(self, 'walker2d.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, action):
        # Move the model
        self.steps_taken += 1
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()

        # Compute the reward
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(action).sum()

        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)

        # Give the reward only every Nth step, or when finished
        if self.steps_taken % self.steps_threshold == 0 or done:
            reward += self.accumulated_reward
            self.accumulated_reward = 0
        else:
            self.accumulated_reward += reward
            reward = 0

        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005,
                                                    high=.005,
                                                    size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005,
                                                    high=.005,
                                                    size=self.model.nv)
        )

        self.steps_taken = 0
        self.accumulated_reward = 0

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
