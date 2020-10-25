import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class DelayedHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, steps_threshold=10):
        self.steps_threshold = steps_threshold
        self.steps_taken = 0
        self.accumulated_reward = 0

        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, action):
        # Move the model
        self.steps_taken += 1
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        ob = self._get_obs()

        # Compute the rewards
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 1.25 * (pos_after - pos_before) / self.dt
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        # Give the reward only every Nth step, or when finished
        if self.steps_taken % self.steps_threshold == 0 or done:
            reward += self.accumulated_reward
            self.accumulated_reward = 0
        else:
            self.accumulated_reward += reward
            reward = 0

        return ob, reward, done, dict(reward_linvel=lin_vel_cost,
                                      reward_quadctrl=-quad_ctrl_cost,
                                      reward_alive=alive_bonus,
                                      reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )

        self.steps_taken = 0
        self.accumulated_reward = 0

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
