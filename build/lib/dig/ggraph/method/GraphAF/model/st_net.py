import numpy as np
import torch
import torch.nn as nn

class ST_Net_Sigmoid(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Sigmoid, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
            self.rescale2 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
            self.rescale2 = Rescale()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
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
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.sigmoid(s + self.sigmoid_shift)
        s = self.rescale2(s) # linear scale seems important, similar to learnable prior..
        return s, t


class ST_Net_Exp(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Exp, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale())
            #self.rescale2 = nn.utils.weight_norm(Rescale())

        else:
            self.rescale1 = Rescale()
            #self.rescale2 = Rescale()

        self.tanh = nn.Tanh()
        #self.sigmoid = nn.Sigmoid()
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
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = self.linear2(self.tanh(self.linear1(x)))
        #x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.rescale1(torch.tanh(s))
        #s = self.sigmoid(s + self.sigmoid_shift)
        #s = self.rescale2(s) # linear scale seems important, similar to learnable prior..
        return s, t


class ST_Net_Softplus(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=64, num_layers=2, bias=True, scale_weight_norm=False, sigmoid_shift=2., apply_batch_norm=False):
        super(ST_Net_Softplus, self).__init__()
        self.num_layers = num_layers  # unused
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.bias = bias
        self.apply_batch_norm = apply_batch_norm
        self.scale_weight_norm = scale_weight_norm
        self.sigmoid_shift = sigmoid_shift

        self.linear1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.linear2 = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.linear3 = nn.Linear(hid_dim, output_dim*2, bias=bias)

        if self.apply_batch_norm:
            self.bn_before = nn.BatchNorm1d(input_dim)
        if self.scale_weight_norm:
            self.rescale1 = nn.utils.weight_norm(Rescale_channel(self.output_dim))

        else:
            self.rescale1 = Rescale_channel(self.output_dim)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        #self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear3.weight, 1e-10)
        if self.bias:
            nn.init.constant_(self.linear1.bias, 0.)
            nn.init.constant_(self.linear2.bias, 0.)
            nn.init.constant_(self.linear3.bias, 0.)

    def forward(self, x):
        '''
        :param x: (batch * repeat_num for node/edge, emb)
        :return: w and b for affine operation
        '''
        if self.apply_batch_norm:
            x = self.bn_before(x)

        x = F.tanh(self.linear2(F.relu(self.linear1(x))))
        x = self.linear3(x)
        #x = self.rescale1(x)
        s = x[:, :self.output_dim]
        t = x[:, self.output_dim:]
        s = self.softplus(s)
        s = self.rescale1(s) # linear scale seems important, similar to learnable prior..
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


class Rescale_channel(nn.Module):
    def __init__(self, num_channels):
        super(Rescale_channel, self).__init__()
        self.num_channels = num_channels
        self.weight = nn.Parameter(torch.zeros([num_channels]))

    def forward(self, x):
        if torch.isnan(torch.exp(self.weight)).any():
            raise RuntimeError('Rescale factor has NaN entries')
            
        x = torch.exp(self.weight) * x
        return x

