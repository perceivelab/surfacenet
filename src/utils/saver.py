import __future__

import subprocess
import sys
from datetime import datetime
from os.path import getmtime
from pathlib import Path
from time import time
from typing import Union

import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

#from utils.misc import image_tensor_to_grid


class TBSaver(object):
    """
    Saver allows for saving and restore networks.
    """

    def __init__(self, base_output_dir: Path, tag: str, args: dict, sub_dirs=('train', 'test')):
        # Create experiment directory
        timestamp_str = datetime.fromtimestamp(
            time()).strftime('%Y-%m-%d_%H-%M-%S')
        self.path = base_output_dir / f'{timestamp_str}_{tag}'
        self.path.mkdir(parents=True, exist_ok=True)

        # TB logs
        self.args = args
        self.writer = SummaryWriter(str(self.path))

        # Create checkpoint sub-directory
        self.ckpt_path = self.path / 'ckpt'
        self.ckpt_path.mkdir(parents=True, exist_ok=True)

        # Create output sub-directories
        self.sub_dirs = sub_dirs
        self.output_path = {}

        for s in self.sub_dirs:
            self.output_path[s] = self.path / 'output' / s

        for d in self.output_path.values():
            d.mkdir(parents=True, exist_ok=False)

        # Dump experiment hyper-params
        with open(self.path / 'hyperparams.txt', mode='wt') as f:
            args_str = [f'{a}: {v}\n' for a, v in self.args.items()]
            args_str.append(f'exp_name: {timestamp_str}\n')
            f.writelines(sorted(args_str))

        # Dump command
        with open(self.path / 'command.txt', mode='wt') as f:
            cmd_args = ' '.join(sys.argv)
            f.write(cmd_args)
            f.write('\n')

        # Dump the `git log` and `git diff`. In this way one can checkout
        #  the last commit, add the diff and should be in the same state.
        for cmd in ['log', 'diff']:
            with open(self.path / f'git_{cmd}.txt', mode='wt') as f:
                subprocess.run(['git', cmd], stdout=f)

    def save_model(self, net: torch.nn.Module, name: str, epoch: int):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Copy to CPU
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        # Save
        torch.save(state_dict, self.ckpt_path / f'{name}_{epoch:05d}.pth')

    def save_checkpoint(self, checkpoint: dict, name: str, epoch: int):
        """
        Save checkpoint.
        """
        torch.save(checkpoint, self.ckpt_path / f'{name}_{epoch:05d}.pth')

    def dump_batch_image(self, image: torch.FloatTensor, epoch: int, split: str, name: str, nrow=10):
        """
        Dump image batch into folder (as grid) and tb
        """

        image = torchvision.utils.make_grid(image, nrow=nrow)

        self.writer.add_image(f'{split}/{name}', image, epoch)

    def dump_batch_video(self, video: torch.FloatTensor, epoch: int, split: str, name: str):
        """
        Dump image batch into folder (as grid) and tb
        """

        self.writer.add_video(f'{split}/{name}', video, epoch)

    def dump_histogram(self, tensor: torch.Tensor, epoch: int, desc: str):
        self.writer.add_histogram(desc, tensor.contiguous().view(-1), epoch)

    def dump_metric(self, value: float, epoch: int, *tags):
        self.writer.add_scalar('/'.join(tags), value, epoch)

    def dump_graph(self, net: torch.nn.Module, image: torch.FloatTensor):
        self.writer.add_graph(net, image)

    @staticmethod
    def load_state_dict(model_path: Union[str, Path], verbose: bool = True):
        """
        Load state dict from pre-trained checkpoint. In case a directory is
          given as `model_path`, the last modified checkpoint is loaded.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise OSError('Please provide a valid path for restoring weights.')

        if model_path.is_dir():
            checkpoint = sorted(model_path.glob('*.pth'), key=getmtime)[-1]
        elif model_path.is_file():
            checkpoint = model_path

        if verbose:
            print(f'Loading pre-trained weight from {checkpoint}...')

        return torch.load(checkpoint)

    def close(self):
        self.writer.close()
