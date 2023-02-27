from __future__ import print_function

from tkinter import N

import numpy as np
import torch
from torch.autograd import Variable

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# ======================================================================================================================
def train_vae(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    # set model in training mode
    model.train()

    # start training
    if args.warmup == 0:
        beta = (
            args.max_beta
        )  #! Changed value of beta to variable max_beta, newly given in experiment.py
    else:
        beta = args.max_beta * epoch / args.warmup
        if beta > args.max_beta:
            beta = args.max_beta
    print("beta: {}".format(beta))

    from time import time

    for batch_idx, (data, target) in enumerate(train_loader):
        # start = time()
        # print(batch_idx)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization
        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # import ipdb

        # ipdb.set_trace()
        # reset gradients
        optimizer.zero_grad()
        # loss evaluation (forward pass)
        loss, RE, KL = model.calculate_loss(x, beta, average=True)
        # print(f"loss: {loss}")
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.data.item()
        train_re += -RE.data.item()
        train_kl += KL.data.item()

        # print("Step Time:", time() - start)

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size

    return model, train_loss, train_re, train_kl
