import torch
from torch import nn

from utils import compute_flat_grad, normal_log_density


class ContinuousPolicy(nn.Module):
    # Deep neural policy network
    def __init__(self, state_dim, action_dim):
        super(ContinuousPolicy, self).__init__()
        self.affine1 = nn.Linear(state_dim[0], 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, action_dim[0])
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim[0]))

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, state, train=True):
        action, action_log_std, action_std = self.forward(state)
        if train:
            action = torch.normal(action, action_std).detach()
        return action

    def get_log_prob(self, x, actions):
        # n * number of stocks
        action_mean, action_log_std, action_std = self.forward(x)
        print('wait what:', action_mean.shape, action_log_std.shape)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        # https://en.wikipedia.org/wiki/Fisher_information
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}

    def Fvp_fim(self, v, states, damping=1e-1):
        # Fisher Vector Product
        # M is the second derivative of the KL distance wrt network output (M*M diagonal matrix compressed into a M*1 vector)
        # mu is the network output (M*1 vector)
        M, mu, info = self.get_fim(states)
        mu = mu.view(-1)
        filter_input_ids = set([info['std_id']])

        t = torch.ones(mu.size(), requires_grad=True, device=mu.device)
        # see https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/
        mu_t = (mu * t).sum()
        Jt = compute_flat_grad(mu_t, self.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
        Jtv = (Jt * v).sum()
        Jv = torch.autograd.grad(Jtv, t)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = compute_flat_grad(mu_MJv, self.parameters(), filter_input_ids=filter_input_ids).detach()
        JTMJv /= states.shape[0]
        std_index = info['std_index']
        JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
        return JTMJv + v * damping

if __name__ == '__main__':
    state_dim, action_dim = (2,), (2,)
    actor = ContinuousPolicy(state_dim, action_dim)
    input_states = torch.randn((1,2))
    input_actions = torch.randn((1,2))
    print(actor.get_log_prob(input_states, input_actions).shape)
