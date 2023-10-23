import numpy as np
import scipy.stats as st


class RunningStat:

    def __init__(self, shape):
        self.shape = shape
        self.expect = np.zeros(shape)
        self.varSum = np.zeros(shape)
        self.count = 0

    def record(self, sample):
        self.count += 1
        diff = sample - self.expect
        self.expect += diff/self.count
        self.varSum += diff * (sample - self.expect)

    def mean(self):
        return self.expect

    def sample_variance(self):
        sampleVariance = np.zeros(self.shape)
        if self.count > 1:
            sampleVariance = self.varSum/(self.count-1)
        return sampleVariance

    def half_window(self, confidence):
        halfWindow = np.zeros(self.shape)
        if self.count > 1:
            sampleVariance = self.varSum/(self.count-1)
            std = np.sqrt(sampleVariance)
            t_crit = np.abs(st.t.ppf((1-confidence)/2, self.count-1))
            halfWindow = t_crit * std/np.sqrt(self.count)
        return halfWindow

    def mean_difference(self, other, confidence):
        meanDiff = self.expect - other.expect
        sampleVar1, sampleSize1 = self.sample_variance(), self.sample_size()
        sampleVar2, sampleSize2 = other.sample_variance(), other.sample_size()
        t_crit = np.abs(st.t.ppf((1-confidence)/2, sampleSize1 + sampleSize2 - 2))
        halfWindow = t_crit * np.sqrt(sampleVar1/sampleSize1 + sampleVar2/sampleSize2)
        return meanDiff, halfWindow

    def sample_size(self):
        return self.count

    def std(self):
        return np.sqrt(self.sample_variance())