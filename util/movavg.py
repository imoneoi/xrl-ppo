import numpy as np

class MovAvg:
    def __init__(self, max_size=100):
        self.maxsize = max_size
        self.cache = np.zeros(max_size)

        self.sum = 0.0
        self.size = 0
        self.pointer = 0

    def push(self, item):
        if self.size == self.maxsize:
            self.sum -= self.cache[self.pointer]
        else:
            self.size += 1

        self.cache[self.pointer] = item
        self.pointer = (self.pointer + 1) % self.maxsize

        self.sum += item

    def get(self):
        if self.size == 0:
            return 0

        return self.sum / self.size