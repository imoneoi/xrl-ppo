import numpy as np
import torch

# Fast check list equal http://stackoverflow.com/q/3844948/
def list_all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)

class Batch(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __len__(self):
        lengths = [len(v) for v in self.__dict__.values()]
        assert list_all_equal(lengths), "All items must have same length"

        return lengths[0] if lengths else 0

    def __getitem__(self, index):
        b = Batch()
        for k in self.__dict__.keys():
            if self.__dict__[k] is not None:
                b.__dict__[k] = self.__dict__[k][index]

        return b

    def to_numpy(self):
        b = Batch()
        for k, v in self.__dict__.items():
            b.__dict__[k] = np.array(v)

        return b

    def split(self, bs, shuffle=False):
        length = len(self)

        if shuffle:
            index = np.random.permutation(length)
        else:
            index = np.arange(length)
        
        #split
        temp = 0
        while temp < length:
            ret = Batch()
            for k, v in self.__dict__.items():
                ret.__dict__[k] = v[index[temp: temp + bs]]

            temp += bs
            yield ret

    def append(self, batch):
        assert isinstance(batch, Batch), 'Only append Batch is allowed!'

        for k in batch.__dict__.keys():
            if batch.__dict__[k] is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = batch.__dict__[k]
            elif isinstance(batch.__dict__[k], list):
                self.__dict__[k] += batch.__dict__[k]
            elif isinstance(batch.__dict__[k], np.ndarray):
                raise NotImplementedError("Concatenating numpy arrays is slow, use python list instead.")
            else:
                s = 'No support for append with type'\
                    + str(type(batch.__dict__[k]))\
                    + 'in class Batch.'
                raise TypeError(s)