import numpy as np

class ActionSpace:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape).astype(np.float32)

    def flatten(self) -> int:
        return int(np.prod(self.shape))
