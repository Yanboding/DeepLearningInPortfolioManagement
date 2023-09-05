import numpy as np

def sharpe(returns, freq=30, rfr=0, eps=1e-8):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def max_drawdown(returns, eps=1e-8):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)


def calculate_transaction_remainder_factor(w1, w0, commission_rate):
    """
    @:param w1: target portfolio vector, first element is cash
    @:param w0: rebalanced last period portfolio vector, first element is cash
    @:param commission_rate: rate of commission fee, proportional to the transaction cost
    """
    mu0 = 1
    # this is a better initialization than the one in paper
    mu1 = 1 - commission_rate * np.sum(np.abs(w1-w0))
    while abs(mu1 - mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
               (2 * commission_rate - commission_rate ** 2) *
               np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
              (1 - commission_rate * w1[0])
    return mu1