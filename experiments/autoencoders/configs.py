from typing import NamedTuple

from torch import nn


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
