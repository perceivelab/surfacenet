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
from utils.losses import msssim_loss, rmse_loss
from utils.utils import make_plot_maps


class Trainer:
    def __init__(self, args, datasets):
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

        # Compute splits names
        splits = list(datasets.keys())

        # Setup data loaders
        loader_train = DataLoader(
            datasets['train']['synth'], batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        loader_test = DataLoader(
            datasets['test']['synth'], batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers)

        self.loaders = {
            'train': {'synth': loader_train},
            'test': {'synth': loader_test},
        }

        if args.train_real:
            self.loaders['train']['real'] = DataLoader(
                datasets['train']['real'], batch_size=args.batch_size,
                shuffle=True, num_workers=args.workers)

            self.loaders['test']['real'] = DataLoader(
                datasets['test']['real'], batch_size=args.batch_size,
                shuffle=False, num_workers=args.workers)


    def train(self):
        # Start training
        for epoch in range(self.args.epochs):
            self.run_epoch(epoch, train=True)

            self.run_epoch(epoch, train=False)

            if epoch % self.args.save_every == 0:
                self.args.saver.save_model(self.net, "net", epoch)


    def run_epoch(self, epoch_idx, train=True):
        if train:
            self.net.train()
        else:
            self.net.eval()

        split = "train" if train else "test"
        # Training loop
        for batch_idx, batch in enumerate(tqdm(self.loaders[split]['synth'])):
            step_idx = epoch_idx * len(self.loaders[split]['synth']) + batch_idx
            losses, maps = self.forward_batch(batch, step_idx, train=train)

            for k, v in losses.items():
                self.args.saver.dump_metric(v, step_idx, split, k)

            if batch_idx % self.args.log_every == 0:
                self.args.saver.dump_batch_image(maps['gen'], step_idx, split, 'gen', nrow=5, do_tb=True)
                self.args.saver.dump_batch_image(maps['gt'], step_idx, split, 'gt', nrow=5, do_tb=True) 


    def forward_batch(self, batch, step_idx, train=True, real=False):
        # Move input data to device
        inputs = batch["Render"].to(self.device)

        # Run model
        outputs = self.net(inputs)

        if not real:
            targets = {key: batch[key].to(self.device) for key in batch.keys()}

            maps = {
                'gen': make_plot_maps(inputs, outputs),
                'gt': make_plot_maps(inputs, targets)
            }

            if train:
                return self.train_batch(inputs, outputs, targets, step_idx), maps
            else:
                return self.eval_batch(outputs, targets), maps
        else:
            raise NotImplementedError("Real data not implemented yet.")

            return self.train_real(inputs, outputs), self.make_plot_maps(inputs, outputs)


    def train_batch(self, inputs, outputs, targets, step_idx):
        # Initialize loss dict
        losses = {}

        loss = 0

        if self.args.train_adversarial:
            real_batch = torch.cat([inputs, *[targets[x]
                                              for x in self.maps]], 1)
            fake_batch = torch.cat(
                [inputs, *[outputs[x].detach() for x in self.maps]], 1)
            
            # Train patch discriminator
            self.__train_discr(self.discriminator,
                               self.optim_patch_discr, real_batch, fake_batch.detach())

            # Train generator
            if step_idx >= self.args.adv_start:
                gen_batch = torch.cat([inputs, *[outputs[x]
                                                for x in self.maps]], 1)

                pred_patch = self.discriminator(gen_batch)

                discr_loss = self.alpha_adv * \
                    self.mse_loss(pred_patch, torch.ones(
                        pred_patch.shape, device=self.device))

                losses[f'adv_loss'] = discr_loss.item()

                loss += discr_loss

        for key in outputs.keys():
            out, tar = outputs[key], targets[key]

            l_pix = l1_loss(out, tar)

            l_msssim = msssim_loss(out, tar)

            l_std = l1_loss(out.std(), tar.std())

            l_map = l_pix + self.alpha_m * l_msssim + l_std

            loss += l_map

            losses[f'{key.lower()}_loss'] = l_map.item()
            losses[f'{key.lower()}_l1'] = l_pix.item()
            losses[f'{key.lower()}_rmse'] = rmse_loss(out, tar)
            losses[f'{key.lower()}_rmse_un'] = rmse_loss((out + 1) / 2, (tar + 1) / 2)
                
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Log losses
        losses[f'loss'] = loss.item()

        return losses
    

    def eval_batch(self, outputs, targets):
        # Initialize loss dict
        losses = {}

        for key in outputs.keys():
            out, tar = outputs[key], targets[key]

            losses[f'{key.lower()}_l1'] = l1_loss(out, tar).item()
            losses[f'{key.lower()}_rmse'] = rmse_loss(out, tar)
            losses[f'{key.lower()}_rmse_un'] = rmse_loss((out + 1) / 2, (tar + 1) / 2)

        return losses
    

    def __train_discr(self, net, optim, real_batch, fake_batch):

        pred_real = net(real_batch)
        loss_real = mse_loss(pred_real, torch.ones(
            pred_real.shape, device=self.device))

        pred_fake = net(fake_batch)
        loss_fake = mse_loss(pred_fake, torch.zeros(
            pred_fake.shape, device=self.device))

        loss_discr = 0.5 * (loss_real + loss_fake)

        optim.zero_grad()
        loss_discr.backward()
        optim.step()

        return loss_discr