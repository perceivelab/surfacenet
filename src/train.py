from pathlib import Path

from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker

from datasets.surface_dataset import PicturesDataset, SurfaceDataset
from trainers.trainer import Trainer
from utils.parser import parse

if __name__ == '__main__':
    # Get params
    args = parse()

    #setup accelerator
    tracker = TensorBoardTracker(run_name=args.tag, logging_dir=args.logdir)
    accelerator = Accelerator(log_with=tracker)
    accelerator.init_trackers("surfacenet")

    # Create datasets
    dset_train = SurfaceDataset(
        Path(args.dataset)/'train',
        args.resize
    )

    dset_test = SurfaceDataset(
        Path(args.dataset)/'test',
        args.resize
    )

    datasets = {
        'train': {'synth': dset_train},
        'test': {'synth': dset_test},
    }

    if args.train_real:
        datasets['train']['real'] = PicturesDataset(
            Path(args.dataset)/'train',
            args.resize
        )
        datasets['test']['real'] = PicturesDataset(
            Path(args.dataset)/'test',
            args.resize
        )

    # Create trainer and start training
    trainer = Trainer(args, accelerator, tracker, datasets)
    trainer.train()

    accelerator.end_training()
