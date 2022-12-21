import argparse
from bisect import bisect_right
import os
import sys
import shutil
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as xforms
import torchvision.datasets as datasets
from tqdm import tqdm

parent = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent not in sys.path:
    sys.path = [parent] + sys.path
import kradagrad
from kradagrad.third_party.resnet_cifar10.trainer import (
    model_names, train, validate, save_checkpoint, AverageMeter, accuracy
)
mf = kradagrad.positive_matrix_functions
import kradagrad.math_utils as mu
from kradagrad.third_party.resnet_cifar10 import resnet

def make_optimizer(params, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params, args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    elif args.optimizer == 'ada':
        optimizer = torch.optim.Adam(
            params, args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps
        )
    elif args.optimizer in ['shampoo', 'kradmm', 'krad']:
        hps = kradagrad.HyperParams(
            matrix_eps=args.eps,  weight_decay=args.weight_decay, graft_type=0, beta2=1,            
            block_size=args.block_size,
            best_effort_shape_interpretation=True,
            inverse_exponent_override=args.inverse_exponent_override,
            preconditioning_compute_steps=mu.roundup(50_000, args.batch_size),
        )
        if args.optimizer == 'shampoo':
            optimizer = kradagrad.Shampoo(params, lr=args.lr, hyperparams=hps, momentum=args.momentum)
        elif args.optimizer == 'kradmm':
            optimizer = kradagrad.KradagradMM(params, lr=args.lr, hyperparams=hps, momentum=args.momentum, debug=args.debug)
        elif args.optimizer == 'krad':
            optimizer = kradagrad.KradagradPP(params, lr=args.lr, momentum=args.momentum, hyperparams=hps,
                                             tensor_batch_size=0)  # tensor batching only for single run
    else:
        raise ValueError('unknown optimizer: {}'.format(args.optimizer))
    return optimizer


def train_run(args):
    start_epoch = 0
    best_prec1 = 0
    
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard_{}{}'.format(args.optimizer, args.opt_modifier_str)))
    

    model = torch.nn.DataParallel(resnet.__dict__[args.arch](activation=args.activation))
    model.cuda()
    
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    ## choose optimizer
    optimizer = make_optimizer(model.parameters(), args)
    milestones = [100, 150]
    if args.optimizer in ['sgd', 'ada']:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, last_epoch=start_epoch - 1
        )
    ## End optimizer setup
    

    # resume from a checkpoint    
    loss_s = []
    prec_s = []
    found = False
    final_path = os.path.join(args.save_dir, 'model_{}{}.th'.format(args.optimizer, args.opt_modifier_str))
    ckpt_path = os.path.join(args.save_dir, 'checkpoint_{}{}.th'.format(args.optimizer, args.opt_modifier_str))
    if os.path.isfile(final_path):
        fn = final_path
        found = True
    elif os.path.isfile(ckpt_path):
        fn = ckpt_path
        found = True
    else:
        found = False
    if found:
        print(" - found checkpoint '{}'".format(fn))
        checkpoint = torch.load(fn)
        start_epoch = checkpoint['epoch']
        loss_fn = os.path.join(args.save_dir, 'loss_{}{}.npy'.format(args.optimizer, args.opt_modifier_str))
        prec_fn = os.path.join(args.save_dir, 'prec_{}{}.npy'.format(args.optimizer, args.opt_modifier_str))
        loss_s = np.load(loss_fn)
        prec_s = np.load(prec_fn)        
        print("     loss '{}'".format(loss_fn))
        print("     prec '{}'".format(prec_fn))
        if start_epoch < args.epochs:
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state'])
            print(" - loaded checkpoint (epoch {})".format(start_epoch))
            loss_s = list(loss_s)
            prec_s = list(prec_s)
        else:
            print(" - model done training")
            return loss_s, prec_s
    else:
        print(" - no checkpoint found in '{}'".format(args.save_dir))

    cudnn.benchmark = True

    normalize = xforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    xforms_tr = [xforms.RandomHorizontalFlip(), xforms.RandomCrop(32, 4), xforms.ToTensor(), normalize]

    dataset_to_use = datasets.CIFAR10 if args.data == 'CIFAR10' else datasets.CIFAR100 if args.data == 'CIFAR100' else None
    dataset_tr = dataset_to_use(root='./data', train=True, transform=xforms.Compose(xforms_tr), download=True)
    dataset_va = dataset_to_use(root='./data', train=False, transform=xforms.Compose([xforms.ToTensor(), normalize]))

    train_loader = torch.utils.data.DataLoader(dataset_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_va, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

    
    if args.evaluate:
        loss_va, prec1_va = validate(val_loader, model, criterion)
        print(loss_va, prec1_va)
        return loss_va, prec1_va

    for epoch in tqdm(range(start_epoch, args.epochs), 'Epoch', ncols=80):
        if args.optimizer in ['shampoo', 'krad', 'kradmm'] and epoch==1:
            print([optimizer.state[x]['preconditioner']._transformed_shape for group in optimizer.param_groups for x in group['params']])
        # train for one epoch
        loss_tr, prec1_tr = train(train_loader, model, criterion, optimizer, epoch, args)

        
        # evaluate on validation set
        loss_va, prec1_va = validate(val_loader, model, criterion, args)
        if args.optimizer in ['sgd', 'ada']:
            lr_scheduler.step()
        #elif args.optimizer in ['shampoo', 'krad', 'kradmm']:
        #    pow_ = bisect_right(milestones, epoch)
        #    for group in optimizer.param_groups:
        #        group['lr'] *= 0.1 ** pow_
        writer.add_scalar('loss/train', loss_tr, epoch)
        writer.add_scalar('loss/val', loss_va, epoch)
        
        writer.add_scalar('prec1/train', prec1_tr, epoch)
        writer.add_scalar('prec1/val', prec1_va, epoch)
        
        if args.optimizer in ['krad', 'kradmm', 'shampoo']:
            precon_norms = {str(i): mf.matrices_norm(precon).cpu().numpy() for i, precon in enumerate([precon_ for group_ in optimizer.param_groups for param_ in group_['params'] for precon_ in optimizer.state[param_]['preconditioner'].preconditioners])}
            writer.add_scalars('norms/precon', precon_norms, epoch)
            stat_norms = {str(i): mf.matrices_norm(stat).cpu().numpy() for i, stat in enumerate([stat_ for group_ in optimizer.param_groups for param_ in group_['params'] for stat_ in optimizer.state[param_]['preconditioner'].statistics])}
            writer.add_scalars('norms/stat', stat_norms, epoch)

        loss_s.append((loss_tr, loss_va))
        prec_s.append((prec1_tr, prec1_va))
        # remember best prec@1 and save checkpoint
        is_best = prec1_va > best_prec1
        best_prec1 = max(prec1_va, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'initial_lr': args.lr,
                'optim_state': optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}{}.th'.format(args.optimizer, args.opt_modifier_str)))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'initial_lr': args.lr,
            'optim_state': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir, 'model_{}{}.th'.format(args.optimizer, args.opt_modifier_str)))
        np.save(os.path.join(args.save_dir, 'loss_{}{}.npy'.format(args.optimizer, args.opt_modifier_str)), np.array(loss_s))
        np.save(os.path.join(args.save_dir, 'prec_{}{}.npy'.format(args.optimizer, args.opt_modifier_str)), np.array(prec_s))
    return loss_s, prec_s

def main(args):
    # does not affect convolutions (cudnn), so should leave resnet alone
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    # add'l arg processing
    args.eps = float(args.eps_str)
    args.lr = float(args.lr_str)
    args.half = not args.not_half

    args.save_dir = '{}_{}_ckpts'.format(args.arch, args.data)

    args.opt_modifier_str = (
        '_eps{}_lr{}_batch{}'.format(
            args.eps_str, args.lr_str, args.batch_size
        )
    ) + (
        '_ONS' if args.inverse_exponent_override else ''
    ) + (
        '_tf32' if args.tf32 else ''
    ) + (
        '_soft' if args.activation=='softplus' else '_relu'
    )

    torch.cuda.set_device(args.device)
    
    
    print("\n=> Training using optimizer '{}{}'".format(args.optimizer, args.opt_modifier_str))

    # reproducible
    torch.manual_seed(4750)
    train_run(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet32',
                       choices=['resnet32', 'resnet56'])
    parser.add_argument('--data', type=str, default='CIFAR10',
                        choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'ada', 'shampoo', 'kradmm', 'krad'])

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--inverse_exponent_override', type=int, default=0)
    parser.add_argument('--eps_str', type=str, default='1e-6')
    parser.add_argument('--lr_str', type=str, default='1e-1')
    parser.add_argument('--not_half', action='store_true')  # not that harmful
    parser.add_argument('--tf32', action='store_true')  # not that helpful
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'softplus'])

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)

