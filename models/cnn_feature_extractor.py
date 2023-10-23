import torch
from torch import nn


class CNNFeatureExtractor(nn.Module):
    # Common feature extractor shared between the state-value V(s) and action-value Q(s,a) function approximations
    def __init__(self, states_dim, kernel_size=3, hidden_channel1=2, hidden_channel2=20):
        super(CNNFeatureExtractor, self).__init__()
        self.eiie_nets = nn.Sequential(
            nn.Conv2d(states_dim[2] - 1, hidden_channel1, kernel_size=(kernel_size, 1), stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channel1, hidden_channel2, kernel_size=(states_dim[0] - kernel_size + 1, 1), stride=1),
            nn.ReLU())
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(hidden_channel2 + 1, 1, kernel_size=(1, 1), stride=1),
            nn.ReLU(),
        )

    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        last_weights, prices = state[:, -1:, -1:, :], state[:, :-1, :, :]
        out = self.eiie_nets(prices)
        out = torch.cat([out, last_weights], dim=1)
        return self.hidden_layer(out).squeeze((1,2))

if __name__ == '__main__':
    fe = CNNFeatureExtractor((50, 6, 4))
    input_samples = torch.randn((10,50, 6, 4))
    print(fe(input_samples).shape)
