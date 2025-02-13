import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

import moco.loader
import moco.builder
from randaugment import rand_augment_transform, GaussianBlur
from losses import PaCoLoss
from models import resnet
from zsl.data.build_zsl import build_dataloader
from scipy.io import savemat, loadmat

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "w")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names += ['resnet200']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet'])
parser.add_argument('--data', metavar='DIR', default='./data')
parser.add_argument('--root_path', type=str, default='./data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:9996', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=8192, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True, type=bool,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True, type=bool,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True, type=bool,
                    help='use cosine lr schedule')
parser.add_argument('--normalize', default=False, type=bool,
                    help='use cosine lr schedule')

# options for paco
parser.add_argument('--mark', default=None, type=str,
                    help='log dir')
parser.add_argument('--reload', default=None, type=str,
                    help='load supervised model')
parser.add_argument('--warmup_epochs', default=10, type=int,
                    help='warmup epochs')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='contrast weight among samples')
parser.add_argument('--beta', default=1.0, type=float,
                    help='contrast weight between centers and samples')
parser.add_argument('--gamma', default=1.0, type=float,
                    help='paco loss')
parser.add_argument('--aug', default=None, type=str,
                    help='aug strategy')
parser.add_argument('--rand_m', default=10, type=int, help='rand aug strategy')
parser.add_argument('--rand_n', default=2, type=int, help='rand aug strategy')

# fp16
parser.add_argument('--fp16', action='store_true', help=' fp16 training')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

best_acc = 0

args = parser.parse_args()
args.DATASETS_SEMANTIC_TYPE="GBU"
args.DATASETS_SEMANTIC = "normalized"
args.DATASETS_NAME = "AWA2"
args.BATCH_SIZE = 128
args.lr = 0.0025 # 5e-5
num_tr_class = 50



folder_name = "./log"
os.makedirs(folder_name, exist_ok=True)

logger_name = "./log/ce_ce_lr" +"_"+str(args.lr)+".mat"
logger = Logger(logger_name)

log_record = str(args)
print(log_record)
logger.write(log_record + '\n')


args.arch='resnet101'
model = getattr(resnet, args.arch)(num_classes=num_tr_class)

# model = moco.builder.MoCo(getattr(resnet, args.arch), args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.normalize)

device = "cuda"
# model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
print(model)

# define loss function (criterion) and optimizer
criterion_ce = nn.CrossEntropyLoss().cuda()
criterion = PaCoLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.moco_t, K=args.moco_k).cuda()

paco_resume = "./checkpoints/paco_r101.pth"
resnet_resume = "YourAccount/projects/InfZSL/pretrained_models/resnet101-5d3b4d8f.pth"

checkpoint = torch.load(resnet_resume)
# for key in checkpoint['state_dict'].keys():
    # print(key)
# for n,p in model.named_parameters():
    # print(n)
checkpoint.pop('fc.weight')
checkpoint.pop('fc.bias')
model.load_state_dict(checkpoint, strict=False)
model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
# optimizer.load_state_dict(checkpoint['optimizer'])
print("Sucessfully load from: ", resnet_resume)

model.fc = torch.nn.Linear(2048, num_tr_class).cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
# cudnn.benchmark = True
tr_dataloader, val_loader,extract_loader, tu_loader, ts_loader, res = build_dataloader(args)

best_val_loss = 1e+12

total_iter_tr = len(tr_dataloader)
total_iter_val = len(val_loader)
total_iter_extract = len(extract_loader)

for epoch in range(30):
    print("epoch: ", epoch)
    # train
    model.train()
    for it_tr, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
        batch_img = batch_img.to(device)
        batch_att = batch_att.to(device)
        batch_label = batch_label.to(device)

        logits_q = model(batch_img)
        ce_loss = criterion_ce(logits_q, batch_label)
        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        if it_tr%10==0:
            log_record = 'Train %d/%d, ce: %.4f' % (it_tr, total_iter_tr, ce_loss.item())
            print(log_record)
            logger.write(log_record + '\n')

    # evals
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    print("start val")
    with torch.no_grad():
        for it_val, (batch_img, batch_att, batch_label) in enumerate(val_loader):
            # print("Val ",it_val, "/", total_iter_val)
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            logits_q = model(batch_img)
            ce_loss = criterion_ce(logits_q, batch_label)
            val_loss+=ce_loss
            acc1 = accuracy(logits_q, batch_label, topk=(1,))[0]
            val_acc+=acc1
        val_loss/=len(val_loader)
        val_acc/=len(val_loader)

        log_record = 'Val ce: %.4f, acc: %.4f' % (val_loss.item(), val_acc.item())
        print(log_record)
        logger.write(log_record + '\n')


        if best_val_loss>val_loss:
            best_val_loss = val_loss

            log_record = 'Best val ce: %.4f' % (best_val_loss)
            print(log_record)
            logger.write(log_record + '\n')


            print("start extract")
            # extract
            feat_record = []
            for it_extract, (batch_img, batch_att, batch_label) in enumerate(extract_loader):
                if it_extract%100==0:
                    print("Extract ",it_extract, "/", total_iter_extract)
                batch_img = batch_img.to(device)
                batch_att = batch_att.to(device)
                batch_label = batch_label.to(device)
                model(batch_img)
                feat = model.module.features
                feat_record.append(feat.cpu())
            feat_record = torch.cat(feat_record,dim=0).numpy()
            print(feat_record.shape)
            save_path = "./out/"+args.dataset+"ce_ce_lr" +"_"+str(args.lr)+".mat"
            savemat(save_path, {'features': feat_record})
            print("Saved at: "+save_path)
