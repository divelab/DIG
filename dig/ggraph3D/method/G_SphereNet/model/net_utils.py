import torch
import torch.nn as nn


class ST_Net_Exp(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)
        self.rescale1 = Rescale()
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear2.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        x = self.linear2(self.tanh(self.linear1(x)))
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        return s, t


class Rescale(nn.Module):
    def __init__(self):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.zeros([1]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            print(self.weight)
            raise RuntimeError('Rescale factor has NaN entries')

        x = torch.exp(self.weight) * x
        return x
    

def init_layer(layer: torch.nn.Linear, w_scale=1.0) -> torch.nn.Linear:
    torch.nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)  # type: ignore
    torch.nn.init.constant_(layer.bias.data, 0)
    return layer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_units=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            init_layer(nn.Linear(input_dim, hidden_units)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_units, 1)),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.layers(x).view(-1)


def flow_reverse(flow_layers, latent, feat):
    for i in reversed(range(len(flow_layers))):
        s, t = flow_layers[i](feat)
        s = s.exp()
        latent = (latent / s) - t
    return latent


def flow_forward(flow_layers, x, feat):
    for i in range(len(flow_layers)):
        s, t = flow_layers[i](feat)
        s = s.exp()
        x = (x + t) * s
        
        if i == 0:
            x_log_jacob = (torch.abs(s) + 1e-20).log()
        else:
            x_log_jacob += (torch.abs(s) + 1e-20).log()
    return x, x_log_jacob