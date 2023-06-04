import argparse
import json
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from kradagrad.utils import get_optimizer
from ray import air, tune
from ray.air import session
from torch.utils.tensorboard import SummaryWriter

from configs import get_task_cfg
from dataloaders import get_dataloaders
from densenet import DenseNet


def main(tune_cfg, args):
    dataset = tune_cfg["dataset"]
    cfg = get_task_cfg(dataset)

    writer = SummaryWriter(log_dir="tb")

    seed = tune_cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net = DenseNet(
        encoder_widths=cfg.encoder_widths,
        decoder_widths=cfg.decoder_widths,
        act_fn=cfg.act_fn,
        out_fn=cfg.out_fn,
    )
    net.cuda()

    train_loader, test_loader = get_dataloaders(
        dataset, batch_size=100, root="/home/luke.walters/data"
    )
    flatten = nn.Flatten()

    opt = tune_cfg["optimizer"].split("_")[0]

    block_size = {"curves": 100, "mnist": 250, "faces": 500}[dataset]
    epochs = {"curves": 150, "mnist": 70, "faces": 150}[dataset]
    optimizer = get_optimizer(
        net.parameters(),
        optimizer=opt,
        lr=tune_cfg["lr"],
        block_size=block_size,
        weight_decay=5e-6,
        double=True if tune_cfg["optimizer"] == "shampoo" else False,
        eps=tune_cfg["eps"],
        iterative_roots=args.iterative,
    )

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

    for epoch in range(epoch_init, epochs):
        for i, batch in enumerate(train_loader, 0):
            if isinstance(batch, list):
                inputs, _ = batch
            else:
                inputs = batch

            if args.show_iter_time:
                start_time = time.time()

            inputs = inputs.float().cuda()
            inputs = flatten(inputs)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = cfg.loss_fn(outputs, inputs)

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

        session.report({"val_loss": val_loss})

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

    parser.add_argument("--status_freq", type=int, default=100)
    parser.add_argument("--from_ckpt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show_iter_time", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--iterative", action="store_true")

    return parser.parse_args()


def get_lr(opt, dataset):
    with open("best_lrs.json") as f:
        best_lrs = json.load(f)

    opt = opt.split("_")[0]
    return best_lrs[dataset][opt]


def get_eps(opt, dataset):
    with open("best_eps.json") as f:
        best_eps = json.load(f)

    opt = opt.split("_")[0]
    return best_eps[dataset][opt]


if __name__ == "__main__":
    args = get_args()

    trainable = partial(main, args=args)

    trainer = tune.with_resources(
        tune.with_parameters(trainable), resources={"cpu": 2, "gpu": 1.0 / 10.0}
    )

    tune_config = tune.TuneConfig(num_samples=1)

    param_space = {
        "optimizer": tune.grid_search(
            ["sgd", "adam", "shampoo", "shampoo_32", "krad*", "krad"]
        ),
        "dataset": tune.grid_search(["curves", "faces", "mnist"]),
        "lr": tune.sample_from(
            lambda spec: get_lr(spec.config.optimizer, spec.config.dataset)
        ),
        "seed": tune.grid_search([1, 22, 333, 4444, 55555]),
        "eps": tune.sample_from(
            lambda spec: get_eps(spec.config.optimizer, spec.config.dataset)
        ),
    }

    exp_name = f"besthps_5seeds"
    if args.tag is not None:
        exp_name += f"_{args.tag}"

    tuner = tune.Tuner(
        trainer,
        param_space=param_space,
        tune_config=tune_config,
        run_config=air.RunConfig(name=exp_name),
    )
    tuner.fit()
