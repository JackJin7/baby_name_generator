from utils import *

class RNNCell(nn.Module):
    def __init__(self, dim, device='cuda', bias=True, dropout=0.25):
        super().__init__()
        self.device = device
        self.dim = dim
        self.hidden_affine = nn.Linear(2 * dim, dim, bias=bias).to(device)  # (2d, d)
        nn.init.xavier_normal_(self.hidden_affine.weight)
        self.output_affine = nn.Linear(dim, dim, bias=bias).to(device)
        nn.init.xavier_normal_(self.output_affine.weight)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x, hx):
        x = torch.cat([x, hx], dim=1)  # (b, 2d)
        hy = self.hidden_affine(x)  # (b, d)
        hy = F.tanh(hy)
        y = self.output_affine(hy)
        y = self.dropout(y)
        return y, hy


class LSTMCell(nn.Module):
    def __init__(self, dim, device='cuda', bias=True, dropout=0.25):
        super().__init__()
        self.device = device
        self.dim = dim
        self.affine = nn.Linear(2 * dim, 4 * dim, bias=bias).to(device)  # (2d, 4d)
        nn.init.xavier_normal_(self.affine.weight)
        self.out_affine = nn.Linear(dim, dim, bias=bias).to(device)
        nn.init.xavier_normal_(self.out_affine.weight)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x, hx, cx):
        x = torch.cat([x, hx], dim=1)  # (b, 2d)

        gates = self.affine(x)
        f, i, c, o = gates.chunk(4, dim=1)

        f = F.sigmoid(f)
        i = F.sigmoid(i)
        c = F.tanh(c)
        o = F.sigmoid(o)

        cy = f * cx + i * c
        hy = o * F.tanh(cy)
        y = self.out_affine(hy)
        y = self.dropout(y)

        return y, hy, cy





