import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class DelayedAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, steps_threshold=10):
        self.steps_threshold = steps_threshold
        self.steps_taken = 0
        self.accumulated_reward = 0

        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        # Move the model
        self.steps_taken += 1
        xposbefore = self.get_body_com('torso')[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com('torso')[0]
        ob = self._get_obs()

        # Compute the rewards
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone

        # Give the reward only every Nth step, or when finished
        if self.steps_taken % self.steps_threshold == 0 or done:
            reward += self.accumulated_reward
            self.accumulated_reward = 0
        else:
            self.accumulated_reward += reward
            reward = 0

        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
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
