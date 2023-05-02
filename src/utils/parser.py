import argparse
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument('--dataset')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--workers', type=int, default=-1, help='-1 for <batch size> threads, 0 for main thread, >0 for background threads')
    # Experiment options
    parser.add_argument('-t', '--tag', default='default')
    parser.add_argument('--logdir', default='exps', type=Path)
    parser.add_argument('-tb', '--tensorboard', action='store_true')
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--grads-hist', action='store_true')
    parser.add_argument('--save-every', type=int, default=100)
    # Training options
    parser.add_argument('--train-real', action='store_true')
    parser.add_argument('--train-adversarial', action='store_true')
    parser.add_argument('--adv-start', type=int, default=300000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

    args = parser.parse_args()
    return args