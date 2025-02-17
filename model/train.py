# Run this file using `python -m model.train`
import argparse

import torch

from .model import Fusion_Model_BLSTM_ATT

description = """
Model training script.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Script to ')

    subcmd = parser.add_subparsers(dest='command', help='sub-command help')
    train_cmd = subcmd.add_parser('train', help='train blstm model')
    train_cmd.add_argument('--epochs', type=int, default=500, help='number of epochs')
    train_cmd.add_argument('--base', type=int, default=0, help='base checkpoint to start training from')
    train_cmd.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    train_cmd.add_argument('--w2v', type=int, default=0, help='w2v checkpoint to generate embeddings from')

    return parser.parse_args()

if __name__ != '__main__':
    print("model/train.py is not meant to be imported, \
            it's meant to be run as a script using \
            `python -m model.train`")
    exit()
    
args = parse_args()
if args.command == 'train':
    num_epochs = args.epochs
    base = args.base
    learning_rate = args.lr
    w2v_cp = args.w2v

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {mps_available}")
    cpu_available = torch.device('cpu')
    print(f"CPU Available: {cpu_available}")
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))

    model = Fusion_Model_BLSTM_ATT(
        w2v_cp,
        device=device,
        mode='train',
        base=base
    )

    model.train_model(
        num_epochs,
        lr=learning_rate
    )
    exit()
