from __future__ import print_function

import argparse
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from filelock import FileLock
from kradagrad.utils import get_optimizer
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from utils.load_data import load_dataset
from utils.optimizer import AdamNormGrad

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #


# Training settings
parser = argparse.ArgumentParser(description="VAE+VampPrior")
# arguments for optimization
parser.add_argument(
    "--batch_size",
    type=int,
    default=200,
    metavar="BStrain",
    help="input batch size for training (default: 200)",
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=1000,
    metavar="BStest",
    help="input batch size for testing (default: 1000)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=400,
    metavar="E",
    help="number of epochs to train (default: 400)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0005,
    metavar="LR",
    help="learning rate (default: 0.0005)",
)
parser.add_argument(
    "--early_stopping_epochs",
    type=int,
    default=50,
    metavar="ES",
    help="number of epochs for early stopping",
)

parser.add_argument(
    "--warmup", type=int, default=100, metavar="WU", help="number of epochs for warm-up"
)
parser.add_argument(
    "--max_beta",
    type=float,
    default=1.0,
    metavar="B",
    help="maximum value of beta for training",
)

# cuda
parser.add_argument(
    "--no-cuda", action="store_true", default=False, help="enables CUDA training"
)
# random seed
parser.add_argument(
    "--seed", type=int, default=14, metavar="S", help="random seed (default: 14)"
)

# model: latent size, input_size, so on
parser.add_argument(
    "--num_layers", type=int, default=1, metavar="NL", help="number of layers"
)

parser.add_argument(
    "--z1_size", type=int, default=200, metavar="M1", help="latent size"
)
parser.add_argument(
    "--z2_size", type=int, default=200, metavar="M2", help="latent size"
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=600,
    metavar="H",
    help="the width of hidden layers",
)
parser.add_argument(
    "--input_size", type=int, default=[1, 28, 28], metavar="D", help="input size"
)

parser.add_argument(
    "--activation", type=str, default=None, metavar="ACT", help="activation function"
)

parser.add_argument(
    "--number_components",
    type=int,
    default=1000,
    metavar="NC",
    help="number of pseudo-inputs",
)
parser.add_argument(
    "--pseudoinputs_mean",
    type=float,
    default=0.05,
    metavar="PM",
    help="mean for init pseudo-inputs",
)
parser.add_argument(
    "--pseudoinputs_std",
    type=float,
    default=0.01,
    metavar="PS",
    help="std for init pseudo-inputs",
)

parser.add_argument(
    "--use_training_data_init",
    action="store_true",
    default=False,
    help="initialize pseudo-inputs with randomly chosen training data",
)

# model: model name, prior
parser.add_argument(
    "--model_name",
    type=str,
    default="baseline",
    metavar="MN",
    help="model name: baseline, vamp, hvamp, hvamp1",
)

parser.add_argument(
    "--input_type",
    type=str,
    default="binary",
    metavar="IT",
    help="type of the input: binary, gray, continuous, multinomial",
)

parser.add_argument(
    "--gated", action="store_true", default=False, help="use gating mechanism"
)

# experiment
parser.add_argument(
    "--S",
    type=int,
    default=5000,
    metavar="SLL",
    help="number of samples used for approximating log-likelihood",
)
parser.add_argument(
    "--MB",
    type=int,
    default=100,
    metavar="MBLL",
    help="size of a mini-batch used for approximating log-likelihood",
)

# dataset
parser.add_argument(
    "--dataset_name",
    type=str,
    default="ml20m",
    metavar="DN",
    help="name of the dataset:  ml20m, netflix, pinterest",
)
parser.add_argument(
    "--optimizer",
    default="sgd",
    const="adam_normgrad",
    nargs="?",
    choices=["sgd", "adam", "shampoo", "krad", "kradmm", "adam_normgrad"],
    help="optimizer",
)

parser.add_argument(
    "--dynamic_binarization",
    action="store_true",
    default=False,
    help="allow dynamic binarization",
)

# note
parser.add_argument(
    "--note",
    type=str,
    default="none",
    metavar="NT",
    help="additional note on the experiment",
)
parser.add_argument(
    "--no_log", action="store_true", default=False, help="print log to log_dir"
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


kwargs = (
    {"num_workers": 0, "pin_memory": True} if args.cuda else {}
)  #! Changed num_workers: 1->0 because of error

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def run(config, args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:10]

    model_name = (
        args.dataset_name
        + "_"
        + args.model_name
        + "_"
        + "(K_"
        + str(args.number_components)
        + ")"
        + "_"
        + args.input_type
        + "_beta("
        + str(args.max_beta)
        + ")"
        + "_layers("
        + str(args.num_layers)
        + ")"
        + "_hidden("
        + str(args.hidden_size)
        + ")"
        + "_z1("
        + str(args.z1_size)
        + ")"
        + "_z2("
        + str(args.z2_size)
        + ")"
    )

    # DIRECTORY FOR SAVING
    args.seed = config["seed"]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    dir = args.model_signature + "_" + model_name + "/"

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print("load data")

    # loading data
    # with FileLock(os.path.expanduser(".datasets.lock")):

    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # CREATE MODEL======================================================================================================
    print("create model")
    # importing model
    if args.model_name == "baseline":
        from models.Baseline import VAE
    elif args.model_name == "vamp":
        from models.Vamp import VAE
    elif args.model_name == "hvamp":
        from models.HVamp import VAE
    elif args.model_name == "hvamp1":
        from models.HVamp_1layer import VAE
    else:
        raise Exception("Wrong name of the model!")

    model = VAE(args)
    if args.cuda:
        model.cuda()

    args.optimizer = config["optimizer"]
    args.lr = config["lr"]
    optimizer = (
        AdamNormGrad(model.parameters(), lr=config["lr"])
        if args.optimizer == "adam_normgrad"
        else get_optimizer(
            model.parameters(),
            optimizer=args.optimizer,
            lr=config["lr"],
            eps=config["eps"],
            double=True if args.optimizer == "shampoo" else False,
            iterative_roots=False,
            block_size=600,
        )
    )

    # ======================================================================================================================
    print(args)
    log_dir = "vae_experiment_log_" + str(os.getenv("COMPUTERNAME")) + ".txt"

    open(log_dir, "a").close()

    # ======================================================================================================================
    print("perform experiment")
    from utils.perform_experiment import experiment_vae

    test_loss, test_re, test_kl, test_ndcg = experiment_vae(
        args,
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        dir,
        log_dir,
        model_name=args.model_name,
    )

    # os.makedirs("my_model", exist_ok=True)
    # torch.save((model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
    # checkpoint = Checkpoint.from_directory("my_model")
    # result = dict(
    #     zip(
    #         ["test_loss", "test_re", "test_kl", "test_ndcg"],
    #         [test_loss, test_re, test_kl, test_ndcg],
    #     )
    # )
    # session.report(result, checkpoint=checkpoint)
    # ======================================================================================================================


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":

    trainable = partial(run, args=args, kwargs=kwargs)

    trainer = tune.with_resources(
        tune.with_parameters(trainable), resources={"cpu": 3, "gpu": 1.0 / 4.0}
    )
    # scheduler = ASHAScheduler(
    #     max_t=args.epochs, grace_period=50, metric="val_loss", mode="min"
    # )

    tune_config = tune.TuneConfig(num_samples=1)

    configs = {}

    for opt in ["krad"]:
        configs[opt] = {
            "optimizer": tune.grid_search([opt]),
            "lr": tune.grid_search([2e-4]),
            "seed": tune.grid_search([200]),
            "eps": 1e-4,
        }

    # for opt in ["adam_normgrad"]:
    #     configs[opt] = {
    #         "optimizer": tune.grid_search([opt]),
    #         "lr": tune.grid_search([4e-4, 2e-4, 1e-4, 5e-5, 2.5e-5, 1.25e-5]),
    #         "seed": tune.grid_search([100]),
    #         "eps": 1e-6,
    #     }

    # results = {}

    if 0:
        for name, params in configs.items():

            tuner = tune.Tuner(trainer, param_space=params, tune_config=tune_config)
            tuner.fit()
    else:
        config = {
            "optimizer": "krad",
            "lr": 2e-4,
            "seed": 200,
            "eps": 1e-4,
        }
        trainable(config)

        # results[name] = tuner.fit()

    # torch.save(results, "ray_tune_results")

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
