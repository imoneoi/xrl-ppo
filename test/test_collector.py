from test_envs.cartpole_continous import CartPoleContinousEnv
from gym.wrappers.time_limit import TimeLimit

from test_policies.pd import PD

from xrl.collector import Collector
from xrl.vectorenv.dummy import VectorEnv
import numpy as np

def MakeEnv():
    return TimeLimit(CartPoleContinousEnv(), max_episode_steps=200)

env = VectorEnv([MakeEnv for _ in range(16)])

PD_coeff = np.array([2.0, 1.0, 10.0, 2.0])
policy = PD(PD_coeff)

collector = Collector(env, policy)

for _ in range(10):
    collector.clear_buffer()
    print(collector.collect(n_step=2001))

    #get batch
    batch = collector.get_experience()

    #validate transition
    n_done = 0
    n_dummy = 0
    length = len(batch)

    for i in range(length):
        trans = batch[i]
        next_trans = batch[i + 1] if i + 1 < length else None

        n_done += trans.done
        n_dummy += trans.dummy

        #State correct
        assert trans.done or (trans.next_state == next_trans.state).all()

        #Action correct
        assert trans.dummy or policy(trans.state) == trans.action

        #Reward correct
        assert trans.reward == 1

        #Dummy insertion correct
        assert (not trans.dummy) or trans.done

    print("dones: {} dummies: {}".format(n_done, n_dummy))