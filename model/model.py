import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time
import datetime

from . import epoch2cp
from . import model_save_path
from . import blstm_default_args as default_args
from .dataloader import CodeDataset

class Fusion_Model_BLSTM_ATT(nn.Module):
    def __init__(
            self,
            # data,
            w2v_cp,
            device=None,
            base=0,
            inference=False
        ):
        """
        data: shape = (batch_size, seq_len, num_channel)
        mode: 'train' or 'eval'
        """

        super().__init__()
        self.device = device

        # hyperparameters
        args = default_args
        self.hidden_size = args['hidden_size']
        self.dropout = args['dropout']

        if not inference:
            self.dataset = CodeDataset(w2v_cp, device, mode='train')
            self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

            self.eval_dataset = CodeDataset(w2v_cp, device, mode='eval')
            self.eval_loader = DataLoader(self.dataset, batch_size=4, shuffle=True)

            # getting dim
            example_batch, _ = next(iter(self.loader))
            dim = example_batch.shape[2]
        dim = 50

        # model
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            # dropout take effect only if num_layers > 1
            # dropout=self.dropout,
            bidirectional=True
        )
        self.tanh = nn.Tanh()
        self.attention = nn.Linear(self.hidden_size*2, 1)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Linear(self.hidden_size*2, 2)

        if base > 0:
            base_path = os.path.join(model_save_path, epoch2cp(base))
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Specified base checkpoint {base_path} not found") 
            self.load_state_dict(torch.load(base_path))

        self.base = base

        self.to(device)

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

    def train_model(self,
                    epoch,
                    lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        print(f"Training started at {datetime.datetime.now()}")
        start_time = time.time()

        for epoch in range(1, epoch+1):
            self.train()
            total_loss = 0
            for x_batch, y_batch in self.loader:
                x_batch = x_batch.float()
                print("x_batch shape: ", x_batch.shape)
                y_batch = y_batch.long()
                optimizer.zero_grad()
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0:
                avg_loss = total_loss / len(self.loader)
                epoch_time = time.time() - start_time
                print(f"Epoch {self.base + epoch}, loss: {avg_loss:.6f}")
                self.eval_model()
                print(f"Time taken for epoch: {epoch_time:.2f} seconds")
                start_time = time.time()
                # saving the model
                model_path = os.path.join(model_save_path, epoch2cp(self.base + epoch))
                torch.save(self.state_dict(), model_path)

    def eval_model(self):
        self.eval()
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            total_loss = 0
            for x_batch, y_batch in self.eval_loader:
                x_batch = x_batch.float()
                y_batch = y_batch.long()
                logits = self(x_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
            avg_loss = total_loss / len(self.eval_loader)
            print(f"Validation loss: {avg_loss:.6f}")

    def predict(self, x):
        self.eval()
        x = x.to(self.device)
        x = x.unsqueeze(0)
        print("shape: ", x.shape)
        with torch.no_grad():
            x = x.float()
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            return probs

# example & debugging
# input is batches of [label, [v1, v2, ..., vn]]
# data = pd.DataFrame(util.generate_data(100, 2))
# data2 = pd.DataFrame(util.generate_data(100, 2))
