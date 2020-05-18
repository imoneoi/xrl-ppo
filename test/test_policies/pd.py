import numpy as np

class PD:
    def __init__(self, coeff):
        self.coeff = coeff

    def __call__(self, state):
        return np.expand_dims(np.sum(state * self.coeff, axis=-1), -1)