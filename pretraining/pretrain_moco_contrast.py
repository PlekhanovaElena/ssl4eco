#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import math
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision

from pretraining.models.moco import loader
from pretraining.models.moco import builder


from torch.utils.tensorboard import SummaryWriter
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
parser.add_argument('--bands', type=str, default='B12',
                    help='bands to process')                    
parser.add_argument('--lmdb', action='store_true',
                    help='use lmdb dataset') 
parser.add_argument('--weights_file', default=None,
                    help='path to CSV with training data weights')
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int, 
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--moco-version',default='v2', type=str,
                    help='v1 or v2 version of moco')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--mode', nargs='*', default=['s2c'])
parser.add_argument('--dtype', type=str, default='uint8')
parser.add_argument('--season', type=str, default='augment')

parser.add_argument('--in_size', type=int, default=224)
parser.add_argument("--is_slurm_job", action='store_true', help="running in slurm")


class EcoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, transform, dtype, band_names, subset=False):
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
    
class TwoCropsTransform_all:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, season='augment'):
        self.base_transform = base_transform
        self.season = season

    def __call__(self, x):

        if self.season=='augment':
            if x.shape[0] > 1:
                season1, season2 = np.random.choice(range(x.shape[0]), replace = False)
            else:
                season1, season2 = np.random.choice(range(x.shape[0]))
                print("Only one image in folder")
        elif self.season=='fixed':
            np.random.seed(42)
            season1 = np.random.choice(range(x.shape[0]))
            season2 = season1
        elif self.season=='random':
            season1 = np.random.choice(range(x.shape[0]))
            season2 = season1

        x1 = np.transpose(x[season1,:,:,:],(1,2,0))
        x2 = np.transpose(x[season2,:,:,:],(1,2,0))

        q = self.base_transform(x1)
        k = self.base_transform(x2)

        return [q, k]

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, season='augment'):
        self.base_transform = base_transform
        self.season = season

    def __call__(self, x):
        if x.shape[0] > 1:
            season1, season2 = np.random.choice(range(x.shape[0]), 2, replace = False)
        else:
            season1, season2 = np.random.choice(range(x.shape[0]), 2)
            print("Only one image in folder")

        x1 = np.transpose(x[season1,:,:,:],(1,2,0))
        x2 = np.transpose(x[season2,:,:,:],(1,2,0))

        q = self.base_transform(x1)
        k = self.base_transform(x2)

        return [q, k]  


def main():

    args = parser.parse_args()

    if args.bands == 'B12':
        args.band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    elif args.bands == 'B12NDVI':
        args.band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDVI']
    elif args.bands == 'B9':
        args.band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'NDVI']
    args.n_channels = len(args.band_names)
    if args.bands == 'B13':    
        args.n_channels = 13
    elif args.bands == 'B3':
        args.n_channels = 3
    

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)



    ### add slurm option ###
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    print(f"Slurm job: {args.is_slurm_job}")
    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
        print(f"args.rank: {args.rank}, args.world_size: {args.world_size}")


    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu


    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create tb_writer
    os.makedirs(args.checkpoints, exist_ok=True)
    tb_writer = SummaryWriter(os.path.join(args.checkpoints,args.log))    
        
    # create model
    print("=> creating model '{}'".format(args.arch))

    # moco-v2
    if args.moco_version == "v2":
        args.mlp = True
        args.moco_t = 0.2
        args.aug_plus = True
        args.cos = True
    print(f"args.moco_t: {args.moco_t}")


    model = builder.MoCo(
        torchvision.models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, bands=args.bands)
    
    print('model created.')

    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        raise NotImplementedError("Please indicate GPU.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # Adjust the learning rate based on the loaded epoch
            adjust_learning_rate(optimizer, args.start_epoch, args)

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    ### load dataset

    from pretraining.models.rs_transforms_float32 import RandomChannelDrop, \
        RandomBrightness, RandomContrast, ToGray
        
    train_transform = cvtransforms.Compose([
        #cvtransforms.Resize(128),
        cvtransforms.RandomResizedCrop(args.in_size, scale=(0.2, 1.)),
        cvtransforms.RandomApply([
            RandomBrightness(0.4),
            RandomContrast(0.4)
        ], p=0.8),
        cvtransforms.RandomApply([ToGray(args.n_channels)], p=0.2),
        cvtransforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        cvtransforms.RandomHorizontalFlip(),       
        cvtransforms.ToTensor()
        #cvtransforms.RandomApply([RandomChannelDrop(min_n_drop=1, max_n_drop=6)], p=0.5),        
        ])
    
    
    train_dataset = EcoDataset(
        root_dir=args.data,
        transform=TwoCropsTransform(train_transform, season = args.season),
        mode=args.mode,
        dtype = args.dtype,
        band_names = args.band_names,
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
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss,top1,top5 = train(train_loader, model, criterion, optimizer, epoch, args)
        if args.rank==0:    
            tb_writer.add_scalar('loss',loss,global_step=epoch,walltime=None)
            tb_writer.add_scalar('acc1',top1,global_step=epoch,walltime=None)
            tb_writer.add_scalar('acc5',top5,global_step=epoch,walltime=None)


        if epoch%10==9:
            if args.rank==0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.checkpoints,'checkpoint_{:04d}.pth.tar'.format(epoch)))
    
    print('Training finished.')
    if args.rank==0:
        tb_writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.cuda()
    model.train()

    end = time.time()

    for i, s2c in enumerate(train_loader):
        images = s2c
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, top1.avg, top5.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

                                                        
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    ss_time = time.time()
    
    main()
    print('total time: %s.' % (time.time()-ss_time))
