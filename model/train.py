# Run this file using `python -m model.train`
import os
import sys
import argparse

import torch

from . import cp2epoch, model_save_path
from .w2v.build_vocab import gen_vocab
from .w2v.model import SkipGram, get_embd_from_sg, get_embd
from .model import Fusion_Model_BLSTM_ATT

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pycmd import process

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

    list_cmd = subcmd.add_parser('list', help='list available checkpoints')

    infer_cmd = subcmd.add_parser('infer', help='infer using model')
    infer_cmd.add_argument("file", type=str, help='.sol file to infer')
    infer_cmd.add_argument('--base', type=int, default=0, help='base checkpoint to start inference from')
    infer_cmd.add_argument('--w2v', type=int, default=0, help='w2v checkpoint to generate embeddings from')
    infer_cmd.add_argument('--slice', type=str, default='ree', help='slice kind, either ree or ts')

    return parser.parse_args()

if __name__ == '__main__':
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
            base=base
        )

        model.train_model(
            num_epochs,
            lr=learning_rate
        )
        exit()

    if args.command == 'list':
        print("Listing available checkpoints")
        cps = []
        for cp in os.listdir(model_save_path):
            epoch = cp2epoch(cp)
            if epoch is not None:
                cps.append(epoch)

        if len(cps) == 0:
            print("No checkpoints found, try training a model first")
        else:
            print("Available checkpoints:")
            for cp in cps:
                print(cp)

        exit()

    if args.command == 'infer':
        base = args.base
        w2v_cp = args.w2v
        fp = args.file
        slice_kind = args.slice

        if not os.path.exists(fp):
            raise FileNotFoundError(f"File {fp} not found")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        mps_available = torch.backends.mps.is_available()
        print(f"MPS Available: {mps_available}")
        cpu_available = torch.device('cpu')
        print(f"CPU Available: {cpu_available}")
        device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))

        process.process_single(filepath=fp, slice_kind=slice_kind)
        files = os.listdir('eval')

        # assert that both files are present
        if not 'sliced.txt' in files:
            raise FileNotFoundError("sliced.txt not found")
        if not 'antlr.txt' in files:
            raise FileNotFoundError("antlr.txt not found")

        code = ''
        sliced_path = os.path.join('eval', 'sliced.txt')
        with open(sliced_path, 'r') as f:
            s_code = f.read().replace('\n', ' ')
            code += s_code
        code += ' '
        antlr_path = os.path.join('eval', 'antlr.txt')
        with open(antlr_path, 'r') as f:
            a_code = f.read().replace('\n', ' ')
            code += a_code

        # w2v_cp_path = os.path.join(w2v_save_path, epoch2cp(w2v_cp))
        vecs = get_embd(code, w2v_cp)

        model = Fusion_Model_BLSTM_ATT(
            w2v_cp,
            device=device,
            base=base
        )

        result = model.predict(vecs)
        print(result)
        exit()

def warmup(
        base: int,
        w2v_cp: int,
    ):
    """
    For efficiency, we pre-load the model and word2vec model
    """
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    device = torch.device('cuda' if cuda_available else ('mps' if mps_available else 'cpu'))
    model = Fusion_Model_BLSTM_ATT(
        w2v_cp,
        device=device,
        base=base,
        inference=True
    )
    vocab, _ = gen_vocab()
    sg = SkipGram(len(vocab), base=w2v_cp)
    return model, sg

def analyze_file(
        model: Fusion_Model_BLSTM_ATT,
        w2v_model: SkipGram,
        fp: str,
        slice_kind: str = 'ree'
    ):
    """
    In here we utilize the model and word2vec model that we loaded using 
    warmup() to analyze
    """
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File {fp} not found")

    process.process_single(filepath=fp, slice_kind=slice_kind)
    files = os.listdir('eval')

    # assert that both files are present
    if not 'sliced.txt' in files:
        raise FileNotFoundError("sliced.txt not found")
    if not 'antlr.txt' in files:
        raise FileNotFoundError("antlr.txt not found")

    code = ''
    sliced_path = os.path.join('eval', 'sliced.txt')
    with open(sliced_path, 'r') as f:
        s_code = f.read().replace('\n', ' ')
        code += s_code
    code += ' '
    antlr_path = os.path.join('eval', 'antlr.txt')
    with open(antlr_path, 'r') as f:
        a_code = f.read().replace('\n', ' ')
        code += a_code

    vecs = get_embd_from_sg(w2v_model, code)

    result = model.predict(vecs)
    return result.tolist()[0][1]
