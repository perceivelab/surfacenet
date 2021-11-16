from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
from models.discriminator import PatchDiscriminator
from models.surfacenet import SurfaceNet
from torch.nn.functional import binary_cross_entropy, l1_loss, mse_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.infinite_dataloader import InfiniteDataLoader
from utils.losses.msssim import msssim_loss


class Trainer:
    def __init__(self, args):
        # Store args
        self.args = args
        self.device = torch.device(args.device)

        # Optimizer params
        optim_params = {'lr': args.lr}
        if args.optim == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif args.optim == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}

        # Create optimizer
        optim_class = getattr(optim, args.optim)

        # Setup models
        self.net = SurfaceNet()
        self.net.to(self.device)
        self.optim = optim_class(params=[param for param in self.net.parameters(
        ) if param.requires_grad], **optim_params)

        if args.train_adversarial:
            self.discr = PatchDiscriminator()
            self.discr.to(self.device)
            self.discr.train()
            self.optim_discr = optim_class(params=[param for param in self.discr.parameters(
            ) if param.requires_grad], **optim_params)

        # Params
        self.alpha_m = 0.88
        self.alpha_adv = 0.15

    def train(self, datasets):
        # Get args
        args = self.args
        saver = args.saver
        log_every = args.log_every
        save_every = args.save_every

        # Compute splits names
        splits = list(datasets.keys())

        # Setup data loaders
        loader_train = InfiniteDataLoader(
            datasets['train']['synth'], batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        loader_test = DataLoader(
            datasets['test']['synth'], batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers)

        loaders = {
            'train': {'synth': loader_train},
            'test': {'synth': loader_test},
        }

        if args.train_real:
            loaders['train']['real'] = DataLoader(
                datasets['train']['real'], batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers)

            loaders['test']['real'] = DataLoader(
                datasets['test']['real'], batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers)

        
                    
