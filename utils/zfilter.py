from utils import RunningStat


class ZFilter:

    def __init__(self, shape):
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.record(x)
        return (x - self.rs.mean()) / (self.rs.std() + 1e-8)
