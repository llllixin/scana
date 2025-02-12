import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, random_split
# import scipy as sp
# import sklearn as sk
# from sklearn.model_selection import train_test_split

import os

# internal
import w2v.build_vocab as build_vocab
import w2v.model as w2v_model
import util

def default_args(args=None):
    if args is None:
        args = {}
    table = {
        'lr': 0.001,
        'epochs': 5,
        'hidden_size': 256,
        'dropout': 0.1
    }
    for key in args:
        table[key] = args[key]
    return table

class Fusion_Model_BLSTM_ATT(nn.Module):
    def __init__(
            self,
            # data_ast,
            # data_codeSlicing,
            data,
            name="",
            device=None,
            args=None
        ):
        """
        data: shape = (batch_size, seq_len, num_channel)
        """

        # data_ast: shape = (batch_size, seq_len, num_channel)
        # data_codeSlicing: shape = (batch_size, seq_len, num_channel)
        # """
        super().__init__()
        # self.data_ast = data_ast
        # self.data_codeSlicing = data_codeSlicing
        self.name = name
        self.device = device

        # hyperparameters
        args = default_args(args)
        self.lr = args['lr']
        self.epochs = args['epochs']
        self.hidden_size = args['hidden_size']
        self.dropout = args['dropout']

        labels = []
        sequences = []
        for label, vec in data:
            labels.append(label)
            sequences.append(vec.detach())
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0).to(self.device)

        # TODO: fixed mx ctx length
        positive_mask = (labels == 1)
        positive_idxs = torch.where(positive_mask)[0]
        negative_idxs = torch.where(~positive_mask)[0]
        # positive_idxs = (labels == 1).nonzero(as_tuple=True)[0]
        # negative_idxs = (labels == 0).nonzero(as_tuple=True)[0]
        if len(positive_idxs) > len(negative_idxs):
            undersampled_negative_idxs = torch.randperm(
                len(negative_idxs), 
                device=self.device
            )[:len(positive_idxs)]
            resampled_idxs = torch.cat([positive_idxs, negative_idxs[undersampled_negative_idxs]])
        else:
            undersampled_positive_idxs = torch.randperm(
                len(positive_idxs),
                device=self.device
            )[:len(negative_idxs)]
            resampled_idxs = torch.cat([positive_idxs[undersampled_positive_idxs], negative_idxs])
        x_resampled = padded_sequences[resampled_idxs]
        y_resampled = labels[resampled_idxs]

        dataset = TensorDataset(x_resampled, y_resampled)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=4, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=4, shuffle=True)
        # self.x_train = x_train.to(torch.float32).to(self.device)
        # self.x_test = x_test.to(torch.float32).to(self.device)
        # self.y_train = y_train.to(torch.float32).to(self.device)
        # self.y_test = y_test.to(torch.float32).to(self.device)

        # model
        self.lstm = nn.LSTM(
            input_size=x_resampled.shape[2],
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout take effect only if num_layers > 1
            # dropout=self.dropout,
            bidirectional=True
        )
        # self.sum = util.Sum(dim=1)
        # self.l1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        # self.leakyr1 = nn.LeakyReLU()
        # self.dropout1 = nn.Dropout(self.dropout)
        # self.l2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        # self.leakyr2 = nn.LeakyReLU()
        # self.dropout2 = nn.Dropout(self.dropout)
        # self.l3 = nn.Linear(self.hidden_size*2, 2)
        self.tanh = nn.Tanh()
        self.attention = nn.Linear(self.hidden_size*2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(self.hidden_size*2, 2)

    def forward(self, x):
        # batch_size, seq_len, 2*hidden_size
        x, (_, _) = self.lstm(x)
        # batch_size, seq_len, 1
        scores = self.attention(x)
        # batch_size, seq_len, 1
        scores = self.tanh(scores)
        # batch_size, seq_len
        scores = scores.squeeze(-1)
        # batch_size, seq_len
        attention_weights = self.softmax(scores)
        # batch_size, seq_len, 1
        weights = attention_weights.unsqueeze(-1)
        # batch_size, 2*hidden_size
        ctx = torch.sum(weights * x, dim=1)
        logits = self.classifier(ctx)
        return logits

        # x = self.sum(x)
        #
        # x = self.l1(x)
        # x = self.leakyr1(x)
        # x = self.dropout1(x)
        #
        # x = self.l2(x)
        # x = self.leakyr2(x)
        # x = self.dropout2(x)
        #
        # x = self.l3(x)
        return x

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.train()
            total_loss = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.float()
                y_batch = y_batch.long()
                optimizer.zero_grad()
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    # def eval_model(self):
    #     self.eval()
    #     with torch.no_grad():
    #         output = self(self.x_test)
    #         _, indices = torch.max(output, 1)
    #         correct = torch.sum(indices == self.y_test.long())
    #         total = self.y_test.size(0)
    #         accuracy = correct.float() / total
    #         print(f"Accuracy: {accuracy}")
    #
    #     return output


# example & debugging
# input is batches of [label, [v1, v2, ..., vn]]
# data = pd.DataFrame(util.generate_data(100, 2))
# data2 = pd.DataFrame(util.generate_data(100, 2))

def gen_data():
    data = []
    kinds = os.listdir('./out')
    for kind in kinds:
        # have vulnerabilities
        dirs = os.listdir(f'./out/{kind}/vul')
        for d in dirs:
            files = os.listdir(f'./out/{kind}/vul/{d}')
            if not 'sliced.txt' in files:
                continue
            if not 'antlr.txt' in files:
                continue
            tokens = []
            with open(f'./out/{kind}/vul/{d}/sliced.txt', 'r') as f:
                code = f.read().replace('\n', ' ')
                for token in build_vocab.tokenize(code):
                    tokens.append(token)
            with open(f'./out/{kind}/vul/{d}/antlr.txt', 'r') as f:
                code = f.read().replace('\n', ' ')
                for token in build_vocab.tokenize(code):
                    tokens.append(token)
            vecs = w2v_model.get_embd(tokens)
            data.append((1, vecs))

        # no vulnerabilities
        dirs = os.listdir(f'./out/{kind}/non_vul')
        for d in dirs:
            files = os.listdir(f'./out/{kind}/non_vul/{d}')
            if not 'sliced.txt' in files:
                continue
            if not 'antlr.txt' in files:
                continue
            tokens = []
            with open(f'./out/{kind}/non_vul/{d}/sliced.txt', 'r') as f:
                code = f.read().replace('\n', ' ')
                for token in build_vocab.tokenize(code):
                    tokens.append(token)
            with open(f'./out/{kind}/non_vul/{d}/antlr.txt', 'r') as f:
                code = f.read().replace('\n', ' ')
                for token in build_vocab.tokenize(code):
                    tokens.append(token)
            vecs = w2v_model.get_embd(tokens)
            data.append((0, vecs))
    return data[:100]

if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    cpu_available = torch.device('cpu')
    print(f"CPU Available: {cpu_available}")
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))

    data = gen_data()

    model = Fusion_Model_BLSTM_ATT(data, device=device)
    model.to(device)

    model.train_model()
    # model.eval_model()
