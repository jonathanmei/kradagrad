import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from configs import get_task_cfg
from dataloaders import get_dataloaders
from densenet import DenseNet


def main(args):
    cfg = get_task_cfg(args.dataset)

    writer = SummaryWriter(log_dir="tb")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not args.cpu:
        torch.cuda.manual_seed(seed)

    net = DenseNet(
        encoder_widths=cfg.encoder_widths,
        decoder_widths=cfg.decoder_widths,
        act_fn=cfg.act_fn,
        out_fn=cfg.out_fn,
    )
    if not args.cpu:
        net.cuda()

    train_loader, test_loader = get_dataloaders(
        args.dataset, args.batch_size, root="./data"
    )

    flatten = nn.Flatten()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.from_ckpt:
        checkpoint = torch.load("best_model.pt")
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_init = checkpoint["epoch"]
    else:
        epoch_init = 0

    running_loss = 0
    train_steps = 0
    best_epoch = 0
    best_val = 100000

    for epoch in range(epoch_init, args.epochs):
        for i, batch in enumerate(train_loader, 0):
            if isinstance(batch, list):
                inputs, _ = batch
            else:
                inputs = batch

            if args.show_iter_time:
                start_time = time.time()

            if not args.cpu:
                inputs = inputs.float().cuda()
            inputs = flatten(inputs)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = cfg.loss_fn(outputs, inputs)
            if torch.isnan(loss):
                raise ValueError("Nan in loss.")

            loss.backward()
            optimizer.step()

            if args.show_iter_time:
                print(f"batch {i}, step time: {time.time()-start_time} sec")

            # print statistics
            running_loss += loss.item()
            if train_steps % args.status_freq == 0:  # print every n mini-batches
                avg_loss = running_loss / args.status_freq
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, avg_loss))

                writer.add_scalar("Loss/train", avg_loss, train_steps)
                running_loss = 0.0

            train_steps += 1

        print("Evaluating...")
        running_val = 0
        for i, batch in enumerate(test_loader, 0):
            if isinstance(batch, list):
                inputs, _ = batch
            else:
                inputs = batch
            inputs = inputs.float().cuda()
            inputs = nn.Flatten()(inputs)
            outputs = net(inputs)
            loss = cfg.loss_fn(outputs, inputs)
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
                "best_model.pt",
            )

    print("Reevaluating best model...")
    checkpoint = torch.load("best_model.pt")
    net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_epoch = checkpoint["epoch"]
    val_loss = checkpoint["loss"]

    running_loss = 0
    for i, batch in enumerate(test_loader, 0):
        if isinstance(batch, list):
            inputs, _ = batch
        else:
            inputs = batch
        inputs = inputs.float().cuda()
        inputs = nn.Flatten()(inputs)
        outputs = net(inputs)
        loss = cfg.loss_fn(outputs, inputs)
        running_loss += loss.item()

    test_loss = running_loss / (i + 1)
    print(f"Best Epoch {epoch+1}, Val Loss: {val_loss}, Best Val Loss: {test_loss}")
    writer.add_scalar("Loss/BestVal", test_loss, best_epoch)

    writer.flush()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["mnist", "faces", "curves"]
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--status_freq", type=int, default=100)
    parser.add_argument("--from_ckpt", action="store_true")
    parser.add_argument("--show_iter_time", action="store_true")
    parser.add_argument("--cpu", action="store_true")


    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
