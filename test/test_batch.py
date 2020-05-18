import numpy as np
from xrl.batch import Batch

def test_batch_split():
    b = Batch(state=list(range(16)), reward=list(range(16)))
    b.action = list(range(16))

    b = b.to_numpy()

    real = [
        Batch(state=np.array([0, 1, 2, 3, 4]), reward=np.array([0, 1, 2, 3, 4]), action=np.array([0, 1, 2, 3, 4])),
        Batch(state=np.array([5, 6, 7, 8, 9]), reward=np.array([5, 6, 7, 8, 9]), action=np.array([5, 6, 7, 8, 9])),
        Batch(state=np.array([10, 11, 12, 13, 14]), reward=np.array([10, 11, 12, 13, 14]), action=np.array([10, 11, 12, 13, 14])),
        Batch(state=np.array([15]), reward=np.array([15]), action=np.array([15])),
    ]

    minibatches = list(b.split(5, shuffle=False))
    for i, mb in enumerate(minibatches):
        for k in real[i].__dict__.keys():
            assert (mb.__dict__[k] == real[i].__dict__[k]).all()

def test_batch_append():
    b = Batch(state=list(range(16)))

    b_new = Batch(state=list(range(16, 32)))

    b.append(b_new)

    assert b.state == list(range(32))

def test_batch_indicies():
    b = Batch(state=np.arange(32), dummy=np.tile([False, True], 16))

    b_clean = b[np.logical_not(b.dummy)]

    assert (b_clean.dummy == False).all()
    assert (b_clean.state == np.arange(0, 32, 2)).all()

test_batch_split()
test_batch_append()
test_batch_indicies()