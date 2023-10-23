import torch
from torch import nn

from models import CNNFeatureExtractor
from utils import compute_flat_grad, normal_log_density

class CNNActor(nn.Module):
    """
    The Actor model takes in a state observation as input and
    outputs an action, which is a continuous value.

    It consists of four fully coonected linear layers with ReLU activation functions and
    a final output layer selects one single optimized action for the state
    """

    def __init__(self, state_dim, action_dim, kernel_size=3, hidden_channel1=2, hidden_channel2=20, feature_extractor=None):
        super(CNNActor, self).__init__()

        if feature_extractor:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = CNNFeatureExtractor(state_dim, kernel_size, hidden_channel1, hidden_channel2)
        self.action_mean = nn.Softmax(dim=-1)

        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim[0]))

    def forward(self, state):
        state = self.feature_extractor(state)
        action_mean = self.action_mean(state)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, state, train=True):
        action, action_log_std, action_std = self.forward(state)
        if train:
            action = self.action_mean(torch.normal(action, action_std)).detach()
        return action

    def get_log_prob(self, states, actions):
        # n * number of stocks
        action_mean, action_log_std, action_std = self.forward(states)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        # https://en.wikipedia.org/wiki/Fisher_information
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        std_id = 0
        for i, (name, param) in enumerate(self.named_parameters()):
            if name == "action_log_std":
                std_id = i
                std_index = param_count
            param_count += param.view(-1).shape[0]
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
    state_dim = (50, 6, 4)
    action_dim = (6,)
    actor = CNNActor(state_dim, action_dim)
    input_states = torch.randn((10, *state_dim))
    input_actions = torch.randn((10, *action_dim))
    print(actor.get_log_prob(input_states, input_actions).shape)