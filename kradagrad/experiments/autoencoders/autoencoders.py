import argparse
import os
import sys
from typing import Callable, List, NamedTuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.expanduser("~/experiments"))

from kradagrad.utils import get_optimizer


class DenseNet(nn.Module):
    def __init__(self, encoder_widths, decoder_widths, act_fn=nn.ReLU(), out_fn=None):
        super(DenseNet, self).__init__()

        assert (
            encoder_widths[-1] == decoder_widths[0]
        ), "encoder output and decoder input dims must match"

        enc_layers = {}
        for k in range(len(encoder_widths) - 1):
            enc_layers[f"enc_layer_{k}"] = nn.Linear(
                encoder_widths[k], encoder_widths[k + 1]
            )
        self.enc_layers = nn.ModuleDict(enc_layers)

        dec_layers = {}
        for k in range(len(decoder_widths) - 1):
            dec_layers[f"dec_layer_{k}"] = nn.Linear(
                decoder_widths[k], decoder_widths[k + 1]
            )
        self.dec_layers = nn.ModuleDict(dec_layers)

        self.act_fn = act_fn
        self.out_fn = out_fn

    def forward(self, x):
        for k in range(len(self.enc_layers)):
            x = self.enc_layers[f"enc_layer_{k}"](x)
            x = x if k == len(self.enc_layers) - 1 else self.act_fn(x)

        for k in range(len(self.dec_layers)):
            x = self.dec_layers[f"dec_layer_{k}"](x)
            if k == len(self.dec_layers) - 1:
                x = x if self.out_fn is None else self.out_fn(x)
            else:
                x = self.act_fn(x)
        return x


class MNISTConfig(NamedTuple):
    encoder_widths = [784, 1000, 500, 250, 30]
    decoder_widths = [30, 250, 500, 1000, 784]
    act_fn = nn.ReLU()
    out_fn = None
    loss_fn = nn.CrossEntropyLoss()


class FACESConfig(NamedTuple):
    encoder_widths = [625, 2000, 1000, 500, 30]
    decoder_widths = [30, 500, 1000, 2000, 625]
    act_fn = nn.ReLU()
    out_fn = None
    loss_fn = nn.MSELoss()


class CURVESConfig(NamedTuple):
    encoder_widths = [784, 400, 200, 100, 50, 25, 6]
    decoder_widths = [6, 25, 50, 100, 200, 400, 784]
    act_fn = nn.ReLU()
    out_fn = None
    loss_fn = nn.CrossEntropyLoss()


def get_task_cfg(dataset):
    if dataset == "mnist":
        return MNISTConfig()
    elif dataset == "faces":
        return FACESConfig()
    elif dataset == "curves":
        return CURVESConfig()
    else:
        raise ValueError(f"Dataset {dataset} is not supported")


def get_dataloaders(dataset, batch_size=100):
    if dataset == "mnist":
        train_val_set = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        train_set, val_set = torch.utils.data.random_split(
            train_val_set, [50000, 10000]
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=2
        )

        test_set = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader, test_loader

    elif dataset == "faces":
        raise NotImplementedError()
    elif dataset == "curves":
        raise NotImplementedError()
    else:
        raise ValueError(f"Dataset {dataset} is not supported")


def main(tune_cfg, args):
    cfg = get_task_cfg(args.dataset)

    writer = SummaryWriter(log_dir="tb")

    net = DenseNet(
        encoder_widths=cfg.encoder_widths,
        decoder_widths=cfg.decoder_widths,
        act_fn=cfg.act_fn,
        out_fn=cfg.out_fn,
    )

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    optimizer = get_optimizer(
        net.parameters(),
        optimizer=tune_cfg["optimizer"],
        lr=tune_cfg["lr"],
        block_size=100,
    )

    if args.from_ckpt:
        checkpoint = torch.load("model.pt")
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_init = checkpoint["epoch"]
    else:
        epoch_init = 0

    running_loss = 0
    train_steps = 0
    best_epoch = 0
    best_val = 1000
    for epoch in range(epoch_init, args.epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = nn.Flatten()(inputs)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = cfg.loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if train_steps % args.status_freq - 1 == 0:  # print every n mini-batches
                avg_loss = running_loss / args.status_freq
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))
                writer.add_scalar("Loss/train", avg_loss, train_steps)
                running_loss = 0.0

            train_steps += 1

        print("Evaluating...")
        running_val = 0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = nn.Flatten()(inputs)
            outputs = net(inputs)
            loss = cfg.loss_fn(outputs, labels)
            running_val += loss.item()
        val_loss = running_val / (i + 1)
        print(f"Epoch {epoch+1}, val loss: {val_loss}")
        writer.add_scalar("Loss/val", val_loss, train_steps)

        if val_loss < best_val:
            best_epoch = epoch
            best_val = val_loss
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val,
                },
                "model.pt",
            )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["mnist", "faces", "curves"]
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--status_freq", type=int, default=100)
    parser.add_argument("--from_ckpt", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    tune_cfg = {"optimizer": "krad", "lr": 0.001}
    main(tune_cfg, args)
