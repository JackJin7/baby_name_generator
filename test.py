import argparse
import networkx as nx
from models import *
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Baby names with RNN.")
    parser.add_argument('--model', default='RNN', help='RNN or LSTM. default RNN')
    parser.add_argument('--device', default='cuda', help='cpu/gpu devices. default gpu')
    parser.add_argument('--dropout', default=0.0, help='Dropout rate. default ')
    return parser.parse_args()


def visualize_gen(pre_name, outputs):  # outputs (t, d)
    outputs = outputs.cpu().detach().numpy()
    G = nx.DiGraph()  # 图
    pos = {}  # 节点位置字典 key:节点ID value:[x,y]
    labels = {}  # 节点label（显示在图上） key:节点ID value:label

    G.add_node(0)
    labels[0] = pre_name
    pos[0] = [0, 0]

    prev_ID = 0
    ID = 1
    pos_y = 0
    for i in range(len(pre_name) - 1, outputs.shape[0]):
        pos_y -= 2
        index = np.argsort(outputs[i])[::-1]
        index = index[:5]
        letters = vec2word(index)
        for j in range(5):
            G.add_node(ID)
            G.add_edge(prev_ID, ID)
            labels[ID] = letters[j]
            pos[ID] = [-2 + j, pos_y]
            ID += 1
        prev_ID = ID - 5

    nx.draw(G, pos=pos, with_labels=False, node_color='#6CB6FF')
    nx.draw_networkx_labels(G, pos, labels=labels)
    plt.show()


def RNN_gen(model, prefix):
    pre_name = prefix
    prefix, _ = letter_transform(prefix)  # (t, d)
    prefix = torch.tensor(prefix).float().to(device)
    prefix = prefix.reshape([1, -1, 28]).permute([1, 0, 2])  # (t, 1, d)

    hidden = torch.zeros(1, prefix.shape[2]).to(device)  # (b, d)
    steps = prefix.shape[0]

    outputs = []
    for t in range(steps):
        output, hidden = model(prefix[t], hidden)  # (b, d), (b, d)
        outputs.append(output)
    for _ in range(steps, 15):
        output = torch.argmax(output, dim=1)
        next_input = torch.zeros([28]).to(device)
        next_input[output] = 1
        next_input = next_input.reshape([1, 28])
        output, hidden = model(next_input, hidden)  # (b, d), (b, d)
        outputs.append(output)
        if torch.argmax(output, dim=1) == 27:
            break

    # output (t, 1, d)
    outputs = torch.stack(outputs, dim=0)
    name = torch.squeeze(outputs)  # (t, d)
    name = torch.argmax(name, dim=1)
    # print(vec2word(name))
    return pre_name + vec2word(name)[steps - 1:]


def LSTM_gen(model, prefix):
    pre_name = prefix
    prefix, _ = letter_transform(prefix)  # (t, d)
    prefix = torch.tensor(prefix).float().to(device)
    prefix = prefix.reshape([1, -1, 28]).permute([1, 0, 2])  # (t, 1, d)

    hidden = torch.zeros(1, prefix.shape[2]).to(device)  # (b, d)
    cell = torch.zeros(1, prefix.shape[2]).to(device)  # (b, d)
    steps = prefix.shape[0]

    outputs = []
    for t in range(steps):
        output, hidden, cell = model(prefix[t], hidden, cell)  # (b, d), (b, d)
        outputs.append(output)
    for _ in range(steps, 15):
        output = torch.argmax(output, dim=1)
        next_input = torch.zeros([28]).to(device)
        next_input[output] = 1
        next_input = next_input.reshape([1, 28])
        output, hidden, cell = model(next_input, hidden, cell)  # (b, d), (b, d)
        outputs.append(output)
        if torch.argmax(output, dim=1) == 27:
            break

    # output (t, 1, d)
    outputs = torch.stack(outputs, dim=0)
    name = torch.squeeze(outputs)  # (t, d)
    # visualize_gen(pre_name, name)
    name = torch.argmax(name, dim=1)
    # print(vec2word(name))
    return pre_name + vec2word(name)[steps - 1:]


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda'

    dropout = args.dropout
    layer_num = args.layer_num

    model_list = ['RNN', 'LSTM']
    epoch_list = [500, 1000, 2000, 3000, 5000]
    # model_list = ['LSTM']
    # epoch_list = [2000]

    models = {}

    if dropout > 0:
        for model in model_list:
            models[model] = {}
            for epoch in epoch_list:
                if model == 'RNN':
                    models[model][epoch] = RNNCell(dim=28, device=device, dropout=dropout)
                else:
                    models[model][epoch] = LSTMCell(dim=28, device=device, dropout=dropout)
                models[model][epoch].load_state_dict(torch.load('model/{}_params_{}.pkl'.format(model, epoch)))
                models[model][epoch].train()
    else:
        for model in model_list:
            models[model] = {}
            for epoch in epoch_list:
                if model == 'RNN':
                    models[model][epoch] = RNNCell(dim=28, device=device)
                else:
                    models[model][epoch] = LSTMCell(dim=28, device=device)
                models[model][epoch].load_state_dict(torch.load('model/{}_params_{}.pkl'.format(model, epoch)))
                models[model][epoch].eval()


    while True:
        prefix = input()
        for model in model_list:
            for epoch in epoch_list:
                if model == 'RNN':
                    name = RNN_gen(models[model][epoch], prefix)
                else:
                    name = LSTM_gen(models[model][epoch], prefix)
                print('{}_{}: {}'.format(model, epoch, name), end='\t')
            print('')


