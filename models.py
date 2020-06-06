from layers import *


class RNN(nn.Module):
    def __init__(self, layer_num, in_dim, out_dim=None, act=lambda x: x, bias=True, dropout=0.25, device='cuda'):
        super().__init__()
        self.act = act
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout).to(device)
        self.layers = nn.ModuleList()

        for i in range(layer_num):
            self.layers.append(RNNLayer(dim=in_dim, device=device, bias=bias, dropout=dropout))

        if self.out_dim:  # 若有out_dim 则再加个1层的MLP
            self.affine = nn.Linear(in_dim, out_dim, bias=bias).to(device)
            nn.init.xavier_normal_(self.affine.weight)


    def forward(self, inputs, length):  # inputs (step, batch, size)  steps=[]
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, length)

        if self.out_dim:
            outputs = self.affine(outputs)

        return self.act(outputs)  # input/output: (t, b, d)


class LSTM(nn.Module):
    def __init__(self, layer_num, in_dim, out_dim=None, act=lambda x: x, bias=True, dropout=0.25, device='cuda'):
        super().__init__()
        self.act = act
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout).to(device)
        self.layers = nn.ModuleList()

        for i in range(layer_num):
            self.layers.append(LSTMLayer(dim=in_dim, device=device, bias=bias, dropout=dropout))

        if self.out_dim:  # 若有out_dim 则再加个1层的MLP
            self.affine = nn.Linear(in_dim, out_dim, bias=bias).to(device)
            nn.init.xavier_normal_(self.affine.weight)

    def forward(self, inputs, length):  # inputs (step, batch, size)  steps=[]
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, length)

        if self.out_dim:
            outputs = self.affine(outputs)

        return self.act(outputs)  # input/output: (t, b, d)
