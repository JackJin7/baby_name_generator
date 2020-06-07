import argparse
from models import *


def parse_args():
    parser = argparse.ArgumentParser(description="Baby names with RNN.")
    parser.add_argument('--model', default='RNN', help='RNN or LSTM. default RNN')
    parser.add_argument('--batch', default=1024, type=int, help='Sampling batch size. default 1024')
    parser.add_argument('--device', default='cuda', help='cpu/gpu devices. default gpu')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate. default 0.1')
    parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate. default 0.1')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs to train. default 1000')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'cpu':
        device = 'cpu'
    else:
        device = 'cuda'

    lr = args.lr
    batch = args.batch
    epochs = args.epochs
    dropout = args.dropout

    X, Y, word_len = load_data()
    loader = {}

    for i in word_len:
        X[i] = torch.tensor(X[i]).float().to(device)
        Y[i] = torch.tensor(Y[i]).float().to(device)
        loader[i] = Data.DataLoader(
            dataset=Data.TensorDataset(X[i], Y[i]),
            batch_size=batch,
            shuffle=True
        )


    if args.model == 'RNN':
        model = RNNCell(dim=28, device=device, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = 0.
            start_time = time()

            for i in word_len:
                for _, (batch_X, batch_Y) in enumerate(loader[i]):

                    batch_X = batch_X.permute(1, 0, 2)
                    batch_Y = batch_Y.permute(1, 0, 2)

                    model.train()
                    model.zero_grad()

                    # (t, b, d)
                    hidden = torch.zeros(batch_X.shape[1], batch_X.shape[2]).to(device)  # (b, d)
                    steps = batch_X.shape[0]

                    outputs = steps * [None]
                    for t in range(steps):
                        outputs[t], hidden = model(batch_X[t], hidden)  # (b, d), (b, d)
                        loss += get_entroy_loss(outputs[t], batch_Y[t])

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss =", "{:.5f}".format(train_loss),
                  "time =", "{:.5f}".format(time() - start_time))


        torch.save(model.state_dict(), 'model/{}_params_{}.pkl'.format(args.model, epochs))

    else:
        model = LSTMCell(dim=28, device=device, dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = 0.
            start_time = time()

            for i in word_len:
                for _, (batch_X, batch_Y) in enumerate(loader[i]):

                    batch_X = batch_X.permute(1, 0, 2)
                    batch_Y = batch_Y.permute(1, 0, 2)

                    model.train()
                    model.zero_grad()

                    hidden = torch.zeros(batch_X.shape[1], batch_X.shape[2]).to(device)  # (b, d)
                    cell = torch.zeros(batch_X.shape[1], batch_X.shape[2]).to(device)  # (b, d)
                    steps = batch_X.shape[0]

                    outputs = steps * [None]
                    for t in range(steps):
                        outputs[t], hidden, cell = model(batch_X[t], hidden, cell)  # (b, d), (b, d)
                        loss += get_entroy_loss(outputs[t], batch_Y[t])

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss =", "{:.5f}".format(train_loss),
                  "time =", "{:.5f}".format(time() - start_time))

        torch.save(model.state_dict(), 'model/{}_params_{}.pkl'.format(args.model, epochs))













