import math


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

if __name__ == '__main__':
    import torch
    x = torch.tensor([[1,1],[0.5,0.5]])
    mean = torch.tensor([[0.6,0.6]])
    log_std = torch.tensor([[0.6,0.6]])
    std = torch.exp(log_std)
    print(normal_log_density(x, mean, log_std, std))