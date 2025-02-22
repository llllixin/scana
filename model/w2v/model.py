import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import w2v_save_path
from . import model_default_args as default_args
from . import epoch2cp
from .build_vocab import gen_vocab, to_ids

# word2vec model, skip-gram
class SkipGram(nn.Module):
    def __init__(self, vocab_size, base=0):
        super(SkipGram, self).__init__()

        embed_size = default_args["embed_size"]
        if embed_size == None or embed_size <= 0:
            raise ValueError("Invalid embed_size, check model/w2v/__init__.py")

        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

        if base > 0:
            base_path = os.path.join(w2v_save_path, epoch2cp(base))
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Specified base checkpoint {base_path} not found") 
            self.load_state_dict(torch.load(base_path))

        self.base = base

    def forward(self, target, pos_context, neg_context):
        v = self.in_embed(target)           # batch_size, embed_size
        pos_u = self.out_embed(pos_context) # batch_size, 2*window_size, embed_size
        neg_u = self.out_embed(neg_context) # batch_size, neg_size, embed_size

        pos_score = torch.sum(torch.bmm(pos_u, v.unsqueeze(2)), dim=2) # batch_size, 2*window_size
        neg_score = torch.sum(torch.bmm(neg_u, v.unsqueeze(2)), dim=2) # batch_size, neg_size

        return pos_score, neg_score

    def loss(self, pos_score, neg_score, mask):
        pos_loss = -F.logsigmoid(pos_score)
        pos_loss = pos_loss * mask
        pos_loss = torch.sum(pos_loss, dim=1) / (torch.sum(mask, dim=1) + 1e-6)
        neg_loss = -F.logsigmoid(-neg_score)

        return pos_loss.mean() + neg_loss.mean()

    def train_sg(self, epoch, target, pos_context, neg_context, mask, lr=1e-4):
        '''
        Args:
            target is a tensor of shape (b, d), 
            pos_context is a matrix of shape (b, n, d) and each row represents a positive context word,
            neg_context is a matrix of shape (b, m, d) and each row represents a negative context word.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr)
        print(f"Training started at {datetime.datetime.now()}")
        start_time = time.time()

        for e in range(1, epoch+1):
            self.train()
            optimizer.zero_grad()
            pos_score, neg_score = self.forward(target, pos_context, neg_context)
            loss = self.loss(pos_score, neg_score, mask)
            loss.backward()
            optimizer.step()

            if (self.base + e) % 100 == 0:
                epoch_time = time.time() - start_time
                print(f"Epoch {self.base + e}, loss: {loss.item()}")
                print(f"Time taken for epoch: {epoch_time:.2f} seconds")
                start_time = time.time()
                model_path = os.path.join(w2v_save_path, epoch2cp(self.base + e))
                torch.save(self.state_dict(), model_path)

def get_embd(code, checkpoint: int):
    vocab, _ = gen_vocab()
    sg = SkipGram(len(vocab), base=checkpoint)
    idx = torch.tensor(to_ids(code))
    embd = sg.in_embed(idx)
    return embd

def get_embd_from_sg(sg: SkipGram, code):
    idx = torch.tensor(to_ids(code))
    embd = sg.in_embed(idx)
    return embd

if __name__ == '__main__':
    print("model/w2v/model.py is not meant to be run as a script")
    exit(1)
