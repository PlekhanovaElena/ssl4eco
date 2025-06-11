#!/usr/bin/env python
import argparse
import os
import random
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pretraining.models.moco import loader
from pretraining.models.moco_v2_seco.moco2_module import MocoV2
from pretraining.models.rs_transforms_float32 import RandomBrightness, \
    RandomContrast, ToGray

from cvtorchvision import cvtransforms
from glob import glob
import rasterio


model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--checkpoints', metavar='DIR', default='./',
                    help='path to checkpoints')
parser.add_argument('--log', metavar='DIR', default='log',
                    help='path to logs')
parser.add_argument('--save-path', metavar='DIR', default='./',
                    help='path to save trained model')
parser.add_argument('--bands', type=str, default='B9',
                    help='bands to process')                    
parser.add_argument('--weights_file', default=None,
                    help='path to CSV with training data weights')
parser.add_argument('--lmdb', action='store_true',
                    help='use lmdb dataset') 
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--subset', action='store_true',
                    help='Should 50k subset be used? e.g. for ablation studies')


# options for moco v2 and seco
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int, 
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--mode', nargs='*', default=['s2c'])
parser.add_argument('--dtype', type=str, default='uint16')
parser.add_argument('--season', type=str, default='augment')

parser.add_argument('--in_size', type=int, default=224)
parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")


class EcoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, transform, dtype, band_names, subset):
        self.dnames = [os.path.join(root_dir, file) for file in os.listdir(root_dir)] #glob(os.path.join(root_dir,'*/*.tif'))
        #self.dnames = [os.path.join(root_dir, dirind) for dirind in os.listdir(root_dir)]
        if subset:
            self.dnames = self.dnames[:50000]
        self.mode = mode
        self.length = len(self.dnames)
        self.transform = transform
        self.dtype = dtype
        self.band_names = band_names
        self.n_channels = len(self.band_names)

        
    def __getitem__(self,index):
        dname = self.dnames[index]
        fnames = glob(os.path.join(dname,'*.tif'))
        
        # Pre-allocate numpy array
        data = np.zeros((len(fnames), self.n_channels, 256, 256), dtype=self.dtype)

        # Read all seasons
        for i, fname in enumerate(fnames):
            with rasterio.open(fname) as rf:
                if self.mode == 'rgb':
                    sdata = rf.read((3,2,1)) # RGB, indeces in .read start with 1
                elif self.band_names == ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'NDVI']:
                    sdata = rf.read((2,3,4,5,6,7,8,9,13)) # 8bands + NDVI
                else:
                    sdata = rf.read()  # Read all bands
                data[i] = sdata

        if self.dtype=='uint8':
            data = (data / 10000 * 255).astype('uint8')
        else:
            data = data.astype('float32')
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    def __len__(self):
        return self.length  

    
class SkipEpochsCallback(Callback):
    def __init__(self, start_epoch):
        self.start_epoch = start_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            # Skip the epoch by setting the flag to True
            trainer.should_stop = True
        else:
            # Continue training normally
            trainer.should_stop = False
            # Print the current learning rate
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    print(f"Epoch {trainer.current_epoch}: Current Learning Rate: {param_group['lr']}")
            
class FourCropsTransform:
    """Take four random crops of one image as the query and key."""
        

    def __init__(self, augment, preprocess):
        self.augment = augment
        self.preprocess = preprocess

    def __call__(self, x):
        n_images = x.shape[0]

        if n_images == 2:
            season1, season2 = np.random.choice(range(n_images), 2, replace = False)
            season3 = season2 
        elif n_images > 2:
            season1, season2, season3 = np.random.choice(range(n_images), 3, replace = False)
        else:
            season1, season2, season3 = np.random.choice(range(n_images), 3)


        t0 = np.transpose(x[season1,:,:,:],(1,2,0))
        t1 = np.transpose(x[season2,:,:,:],(1,2,0))
        t2 = np.transpose(x[season3,:,:,:],(1,2,0))

        q = t0
        k0 = self.augment(t1)
        k1 = t2
        k2 = self.augment(t0)

        q = self.preprocess(q)
        k0 = self.preprocess(k0)
        k1 = self.preprocess(k1)
        k2 = self.preprocess(k2)

        return q, [k0, k1, k2]

def get_experiment_name(hparams):
    data_name = os.path.basename(hparams.data)
    name = f'_{data_name}_b{hparams.batch_size}_epochs={hparams.epochs}_'
    return name

def main():

    args = parser.parse_args()
    args.gpu = 0
    if args.bands == 'B12':
        args.band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    elif args.bands == 'B12NDVI':
        args.band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDVI']
    elif args.bands == 'B9':
        args.band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'NDVI']
    args.n_channels = len(args.band_names)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)


    ### add slurm option ###
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    print(f"Slurm job: {args.is_slurm_job}")
    
    emb_spaces_loc = 3


    # create model
    print("=> creating model '{}'".format(args.arch))

    model = MocoV2(**vars(args), emb_spaces=emb_spaces_loc) #emb_spaces=datamodule.num_keys
    print('model created.')
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    start_epoch = 0
    ckpt_path = None

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            ckpt_path = args.resume
            model = model.load_from_checkpoint(args.resume)
            start_epoch = model.current_epoch
            print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, model.current_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

        

    logger = TensorBoardLogger(
            save_dir=os.path.join(os.getcwd(), 'logs_seco', args.log),
            name=get_experiment_name(args)
        )
    
    
    print("Defining trainer")


    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.epochs,
        resume_from_checkpoint=args.resume,
        enable_model_summary=True,
        enable_progress_bar=True,
        log_every_n_steps=50,
        callbacks=[ModelCheckpoint(filename='{epoch}', every_n_epochs = 10), SkipEpochsCallback(start_epoch=start_epoch)],
        benchmark = True, # speeds up training if the input size stays the same
        accelerator="gpu",
        devices=1
    )


    ### load dataset
    
    

    tr_augment = cvtransforms.Compose([
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(args.n_channels)], p=0.2),
        cvtransforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),           
        ])

    tr_preprocess = cvtransforms.Compose([
        cvtransforms.RandomResizedCrop(args.in_size, scale=(0.2, 1.)),
        cvtransforms.ToTensor()       
        ])
    

    train_dataset = EcoDataset(
        root_dir=args.data,
        transform=FourCropsTransform(tr_augment, tr_preprocess),
        mode=args.mode,
        dtype = args.dtype,
        band_names = args.band_names,
        subset = args.subset
    )   
    print(f"len train dataset: {len(train_dataset)}")

    if args.weights_file is None:
        weights = [1.0] * len(train_dataset)  # Default to equal weights if none provided
    else:
        weights = np.loadtxt(args.weights_file, delimiter=',')
        weights = weights[:len(train_dataset)]
        print(f"Loaded {len(weights)} weights")
        assert len(weights) == len(train_dataset), "Length of weights must match length of dnames"
    
    train_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)    
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.is_slurm_job, sampler=train_sampler, drop_last=True)

    print('start training...')
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    ss_time = time.time()
    
    main()
    print('total time: %s.' % (time.time()-ss_time))
