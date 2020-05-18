import torch
import numpy as np
from torch import nn

class Network(nn.Module):
    def __init__(self, state_shape, action_shape, max_action, device):
        super().__init__()
        self.device = device

        self.max_action = max_action

        def LinearLayer(*args, **kwargs):
            linear = nn.Linear(*args, **kwargs)

            #torch.nn.init.orthogonal_(linear.weight)
            #torch.nn.init.zeros_(linear.bias)
            return linear

        num_hidden = 128
        self.policy_net = nn.Sequential(*[
            LinearLayer(np.prod(state_shape), num_hidden),
            nn.ReLU(inplace=True),
            LinearLayer(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
            LinearLayer(num_hidden, np.prod(action_shape))
        ])

        self.logstd_param = nn.Parameter(torch.zeros( np.prod(action_shape) ))

        self.value_net = nn.Sequential(*[
            LinearLayer(np.prod(state_shape), num_hidden),
            nn.ReLU(inplace=True),
            LinearLayer(num_hidden, num_hidden),
            nn.ReLU(inplace=True),
            LinearLayer(num_hidden, 1)
        ])

        print("Using two networks :)")

    def forward(self, state, inference_policy=True, estimate_value=False):
        state = state.view(state.shape[0], -1)

        if inference_policy:
            #policy_mean = torch.tanh(self.policy_net(state)) * self.max_action
            policy_mean = self.policy_net(state)
            policy_std  = torch.exp(self.logstd_param)

            # # !!! Keep shape same !!!
            policy_std = policy_std + torch.zeros_like(policy_mean)

            policy = torch.distributions.Normal(policy_mean, policy_std)
        else:
            policy = None

        if estimate_value:
            value  = self.value_net(state).squeeze(-1)
        else:
            value = None

        return policy, value