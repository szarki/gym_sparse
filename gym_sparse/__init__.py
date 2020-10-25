from gym.envs.registration import register

register(
    id='SparseHalfCheetah-v0',
    entry_point='gym_sparse.envs:SparseHalfCheetahEnv',
)

register(
    id='DelayedAnt-v0',
    entry_point='gym_sparse.envs:DelayedAntEnv',
)

register(
    id='DelayedHalfCheetah-v0',
    entry_point='gym_sparse.envs:DelayedHalfCheetahEnv',
)

register(
    id='DelayedHopper-v0',
    entry_point='gym_sparse.envs:DelayedHopperEnv',
)

register(
    id='DelayedHumanoid-v0',
    entry_point='gym_sparse.envs:DelayedHumanoidEnv',
)

register(
    id='DelayedWalker2d-v0',
    entry_point='gym_sparse.envs:DelayedWalker2dEnv',
)
