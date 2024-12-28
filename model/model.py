import torch
import torch.nn as nn
import numpy as np
import pandas as pd
# import scipy as sp
# import sklearn as sk
from sklearn.model_selection import train_test_split

# internal
import util

def default_args(args=None):
    if args is None:
        args = {}
    table = {
        'lr': 0.001,
        'epochs': 1,
        'hidden_size': 256,
        'dropout': 0.1
    }
    for key in args:
        table[key] = args[key]
    return table

class Fusion_Model_BLSTM_ATT(nn.Module):
    def __init__(
            self,
            data_ast,
            data_codeSlicing,
            name="",
            device=None,
            args=None
        ):
        """
        data_ast: shape = (batch_size, seq_len, num_channel)
        data_codeSlicing: shape = (batch_size, seq_len, num_channel)
        """
        super().__init__()
        self.data_ast = data_ast
        self.data_codeSlicing = data_codeSlicing
        self.name = name
        self.device = device

        # hyperparameters
        args = default_args(args)
        self.lr = args['lr']
        self.epochs = args['epochs']
        self.hidden_size = args['hidden_size']
        self.dropout = args['dropout']

        # train and test data
        # =================================
        # ast data
        labels_ast = data_ast.iloc[:, 0].values
        vectors_ast = np.stack(data_ast.iloc[:, 1].values)
        positive_idxs_ast = np.where(labels_ast == 1)[0]
        negative_idxs_ast = np.where(labels_ast == 0)[0]

        undersampled_negative_idxs_ast = np.random.choice(negative_idxs_ast, len(positive_idxs_ast), replace=False)
        resampled_idxs_ast = np.concatenate([positive_idxs_ast, undersampled_negative_idxs_ast])
        x_train_ast,x_test_ast,y_train_ast,y_test_ast = \
            train_test_split(
                vectors_ast[resampled_idxs_ast], 
                labels_ast[resampled_idxs_ast], 
                train_size=0.8, test_size=0.2,
            )

        # =================================
        # code slices data
        vectors_slicing = np.stack(data_codeSlicing.iloc[:, 1].values)
        labels_slicing = data_codeSlicing.iloc[:, 0].values
        positive_idxs_slicing = np.where(labels_slicing == 1)[0]
        negative_idxs_slicing = np.where(labels_slicing == 0)[0]

        undersampled_negative_idxs_slicing = np.random.choice(negative_idxs_slicing, len(positive_idxs_slicing), replace=False)
        resampled_idxs_slicing = np.concatenate([positive_idxs_slicing, undersampled_negative_idxs_slicing])
        x_train_slicing,x_test_slicing,y_train_slicing,y_test_slicing = \
            train_test_split(
                vectors_slicing[resampled_idxs_slicing], 
                labels_slicing[resampled_idxs_slicing], 
                train_size=0.8, test_size=0.2,
            )

        x_train = np.concatenate((x_train_ast, x_train_slicing))
        x_test = np.concatenate((x_test_ast, x_test_slicing))
        y_train = np.concatenate((y_train_ast, y_train_slicing))
        y_test = np.concatenate((y_test_ast, y_test_slicing))
        # torch-ify the data
        self.x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)

        # model
        self.lstm = nn.LSTM(
            input_size=self.x_train.shape[2],
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout take effect only if num_layers > 1
            # dropout=self.dropout,
            bidirectional=True
        )
        self.sum = util.Sum(dim=1)
        self.l1 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.leakyr1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.l2 = nn.Linear(self.hidden_size*2, self.hidden_size*2)
        self.leakyr2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.l3 = nn.Linear(self.hidden_size*2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, (_, _) = self.lstm(x)
        x = self.sum(x)

        x = self.l1(x)
        x = self.leakyr1(x)
        x = self.dropout1(x)

        x = self.l2(x)
        x = self.leakyr2(x)
        x = self.dropout2(x)

        x = self.l3(x)
        return x

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.train()
            optimizer.zero_grad()
            output = self(self.x_train)
            loss = criterion(output, self.y_train.long())
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

    def eval_model(self):
        self.eval()
        with torch.no_grad():
            output = self(self.x_test)
            _, indices = torch.max(output, 1)
            correct = torch.sum(indices == self.y_test.long())
            total = self.y_test.size(0)
            accuracy = correct.float() / total
            print(f"Accuracy: {accuracy}")

        return output


# example & debugging
data = pd.DataFrame(util.generate_data(100, 2))
data2 = pd.DataFrame(util.generate_data(100, 2))

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
mps_available = torch.backends.mps.is_available()
print(f"MPS Available: {mps_available}")
cpu_available = torch.device('cpu')
print(f"CPU Available: {cpu_available}")
device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))

model = Fusion_Model_BLSTM_ATT(pd.DataFrame(data), pd.DataFrame(data2), device=device)
model.to(device)

model.train_model()
model.eval_model()
