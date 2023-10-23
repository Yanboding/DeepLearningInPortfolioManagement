import numpy as np
import torch.nn as nn
import torch

from utils import compute_flat_grad


class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_num, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_num = action_num
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = int(np.prod(state_dim))
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, states, action_masks=None):
        if action_masks is None:
            action_masks = torch.tensor([True])
        for affine in self.affine_layers:
            states = self.activation(affine(states))
        # https://ai.stackexchange.com/questions/2980/how-should-i-handle-invalid-actions-when-using-reinforce#:~:text=An%20experimental%20paper,experience%20memory%20too.
        # a trick to set the probability of invalid actions to 0
        action_prob = torch.softmax(self.action_head(states) + torch.log(action_masks + 1e-9), dim=1)
        return action_prob

    def select_action(self, states, action_masks=None, train=True):
        action_prob = self.forward(states, action_masks)
        if train:
            action = action_prob.multinomial(1)
        else:
            action = torch.argmax(action_prob, dim=1, keepdim=True)
        return action

    def get_log_prob(self, states, actions, action_masks=None):
        action_prob = self.forward(states, action_masks)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, states, action_masks=None):
        action_prob = self.forward(states, action_masks)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

    def Fvp_fim(self, v, states, action_masks=None, damping=1e-1):
        M, mu, info = self.get_fim(states, action_masks)
        mu = mu.view(-1)
        filter_input_ids = set()

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
        return JTMJv + v * damping
