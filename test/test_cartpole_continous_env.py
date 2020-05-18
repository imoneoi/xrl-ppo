from test_envs.cartpole_continous import CartPoleContinousEnv
from gym.wrappers.time_limit import TimeLimit

from test_policies.pd import PD
import numpy as np

import time

PD_coeff = np.array([2.0, 1.0, 10.0, 2.0])
policy = PD(PD_coeff)

env = TimeLimit(CartPoleContinousEnv(), max_episode_steps=200)
state = env.reset()

total_reward = total_length = 0

for step in range(10000):
    action = policy(np.array(state))

    state, reward, done, info = env.step(action)
    total_reward += reward
    total_length += 1

    # env.render()
    # time.sleep(0.01)

    if done:
        state = env.reset()

        assert total_reward == 200
        assert total_length == 200

        print("reward: {}, len: {}".format(total_reward, total_length))
        total_reward = total_length = 0