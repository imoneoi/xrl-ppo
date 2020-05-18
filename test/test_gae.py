from xrl.network import Network
from xrl.ppo import PPO
import numpy as np
from xrl.batch import Batch

batch = Batch(
    done=np.array([0, 0, 0, 1., 0, 0, 0, 1, 0, 0, 0, 1]),
    reward=np.array([101, 102, 103., 200, 104, 105, 106, 201, 107, 108, 109, 202]),
    value=np.roll(np.array([2., 3., 4, -1, 5., 6., 7, -2, 8., 9., 10, -3]), 1),
    dummy=np.tile([False], 12)
)

policy = PPO(None, None, gamma=0.99, gae_lambda=0.95, normalize=False)

policy.calculate_gae(batch)

returns = np.array([
    454.8344, 376.1143, 291.298, 200.,
    464.5610, 383.1085, 295.387, 201.,
    474.2876, 390.1027, 299.476, 202.])

assert abs(batch.returns - returns).sum() <= 1e-3

batch = Batch(
    done=np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    reward=np.array([101, 102, 103., 200, 9999, 104, 105, 106, 201, 8888, 107, 108, 109, 202, 6666]),
    #value=np.array([88888, 2., 3., 4, -1, 33333, 5., 6., 7, -2, 44444, 8., 9., 10, -3]),
    value=np.array([3., 4, -1, 8, 33333., 5., 6., 7, -2, 9, 8., 9., 10, -3, 6]),
    dummy=np.array([False, False, False, False, True, False, False, False, False, True, False, False, False, False, True])
)

policy.calculate_gae(batch)

batch = batch[np.logical_not(batch.dummy)]

returns = np.array([
    27907.68132875, 29565.63883972, 31327.685635, 33199.67,
    471.67533043, 390.62023438, 303.321355  , 209.91,
    478.75596778, 394.80113534, 304.41907   , 207.94 ])

assert abs(batch.returns - returns).sum() <= 1e-3