from torch import nn


class ChannelMLP(nn.Module):
    """Channel MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super(ChannelMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, input_shape):
        _, N, C = input_shape
        flops = 0
        flops += (C + 1) * self.hidden_features * N
        flops += (self.hidden_features + 1) * self.out_features * N
        return flops