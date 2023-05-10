from datasets.surface_dataset import PicturesDataset, SurfaceDataset
from trainers.trainer import Trainer
from utils.parser import parse
from utils.saver import TBSaver

if __name__ == '__main__':
    # Get params
    args = parse()

    # Load dataset
    dset_train = SurfaceDataset(
        args.dataset/'train',
        args.resize
    )

    dset_test = SurfaceDataset(
        args.dataset/'test',
        args.resize
    )

    datasets = {
        'train': {'synth': dset_train},
        'test': {'synth': dset_test},
    }

    if args.train_real:
        datasets['train']['real'] = PicturesDataset(
            args.dataset/'train',
            args.resize
        )
        datasets['test']['real'] = PicturesDataset(
            args.dataset/'test',
            args.resize
        )

    # Define saver
    if args.tensorboard:
        saver = TBSaver(args.logdir, args.tag, args.__dict__, sub_dirs=list(
            datasets.keys()))
        # Add saver to args (e.g. visualizing weights, outputs)
        args.saver = saver

    #net = SurfaceNet()
    trainer = Trainer(args, datasets)

    trainer.train()

    if args.tb:
        # Close saver
        saver.close()
