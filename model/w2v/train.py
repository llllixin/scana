# Run this file using `python -m model.w2v.train [command]`
import os
import argparse

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, random_split

# from .model import Fusion_Model_BLSTM_ATT
from . import w2v_save_path
from . import cp2epoch
from . import model_default_args as default_args
from .model import SkipGram
from .build_vocab import gen_vocab, tokenize

description = """
Model training script.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train w2v model')

    subcmd = parser.add_subparsers(dest='command', help='{sub-command} help')

    list_cmd = subcmd.add_parser('list', help='list available checkpoints')

    train_cmd = subcmd.add_parser('train', help='train w2v model')
    train_cmd.add_argument('--epochs', type=int, default=500, help='number of epochs')
    train_cmd.add_argument('--base', type=int, default=0, help='base checkpoint to start training from')
    train_cmd.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    return parser.parse_args()

if __name__ != '__main__':
    print("model/w2v/train.py is not meant to be imported",
          "it's meant to be run as a script using",
          "`python -m model.w2v.train`")
    exit()
    
args = parse_args()
if args.command == 'list':
    cps = os.listdir(w2v_save_path)
    if len(cps) == 0:
        print("No available w2v models, consider",
              "running `python -m model.w2v.train`",
              "to train w2v model first")
    else:
        print("All available w2v models: ")
        for cp in cps:
            print(cp2epoch(cp))
    exit()

if args.command == 'train':
    num_epochs = args.epochs
    base = args.base
    learning_rate = args.lr

    if not os.path.exists("out"):
        raise FileNotFoundError("Directory out not found, preprocess the code files first")

    sentences = []
    for root, _, files in os.walk("out"):
        for file in files:
            if file != "sliced.txt" and file != "antlr.txt":
                continue
            with open(os.path.join(root, file), "r") as f:
                lines = f.readlines()
                for line in lines:
                    sentences.append(line[:-1] if line[-1] == '\n' else line)

    # print("sentences:")
    # print(sentences)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    vocab, word2idx = gen_vocab()

    window_size = default_args["window_size"]
    neg_size = default_args["neg_size"]
    if window_size < 1:
        raise ValueError("Window size should be greater than 0")
    if neg_size < 1:
        raise ValueError("Negative size should be greater than 0")
    targets = []
    pos_context = []
    neg_context = []
    mask = []
    for s in sentences:
        tokens = tokenize(s)
        for i, word in enumerate(tokens):
            target = word2idx[word]
            targets.append(target)
            pos = [word2idx[tokens[j]] for j in range(max(i-window_size, 0), min(i+window_size+1, len(tokens))) if j != i]
            pad_len = 2*window_size - len(pos)
            if pad_len > 0:
                pos += [0] * pad_len
                mask.append([1] * (2*window_size - pad_len) + [0] * pad_len)
            else:
                mask.append([1] * 2*window_size)

            pos_context.append(pos)
            neg = torch.randint(0, len(vocab), (neg_size,)).tolist()
            neg_context.append(neg)

        # TODO: hack
        if len(targets) > 25000:
            break

    print("targets:", targets)
    print("num_classes:", len(vocab))
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    pos_context = torch.tensor(pos_context, dtype=torch.long).to(device)
    neg_context = torch.tensor(neg_context, dtype=torch.long).to(device)
    mask = torch.tensor(mask, dtype=torch.float).to(device)

    # print("targets:", targets.shape)
    # print(targets)
    # print("pos_context:", pos_context.shape)
    # print(pos_context)
    # print("neg_context:", neg_context.shape)
    # print(neg_context)

    sg = SkipGram(len(vocab), base=base).to(device)
    sg.train_sg(num_epochs, targets, pos_context, neg_context, mask, learning_rate)
    exit()
