import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, shape, mu, sigma, theta=.15, dt=1e-2, x0=None, random_seed=42):
        self.shape = shape
        self.theta = theta
        self.mu = np.ones(shape)*mu
        self.sigma = np.ones(shape)*sigma
        self.dt = dt
        self.x0 = x0
        self.np_random = np.random.default_rng(random_seed)
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * self.np_random.normal(size=self.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)