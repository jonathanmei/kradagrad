import os
import sys

import avalanche as avl
import torch
from avalanche.evaluation import metrics as metrics
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.supervised.lamaml import LaMAML
from experiments.utils import create_default_args, set_seed
from models.models_lamaml import MTConvCIFAR
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

sys.path.append(os.path.expanduser("~/"))

import kradagrad


def get_optimizer(
    params,
    optimizer="sgd",
    block_size=128,
    weight_decay=0.0,
    inverse_exponent_override=0,
    eps=1e-6,
    lr=0.01,
    momentum=0.9,
    debug=False,
    precon_update_freq=20,
):
    hps = kradagrad.HyperParams(
        matrix_eps=eps,
        weight_decay=weight_decay,
        graft_type=0,
        beta2=1,
        block_size=block_size,
        best_effort_shape_interpretation=True,
        inverse_exponent_override=inverse_exponent_override,
        preconditioning_compute_steps=precon_update_freq,
    )
    if optimizer == "shampoo":
        optimizer = kradagrad.Shampoo(params, lr=lr, hyperparams=hps, momentum=momentum)
    elif optimizer == "kradmm":
        optimizer = kradagrad.KradagradMM(
            params,
            lr=lr,
            hyperparams=hps,
            momentum=momentum,
            debug=debug,
        )
    elif optimizer == "krad":
        optimizer = kradagrad.KradagradPP(
            params,
            lr=lr,
            momentum=momentum,
            hyperparams=hps,
            tensor_batch_size=0,
        )  # tensor batching only for single run

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params, lr, weight_decay=weight_decay, momentum=momentum
        )
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay, eps=eps)

    return optimizer


def lamaml_scifar100(config):
    """
    "La-MAML: Look-ahead Meta Learning for Continual Learning",
    Gunshi Gupta, Karmesh Yadav, Liam Paull;
    NeurIPS, 2020
    https://arxiv.org/abs/2007.13904
    """
    # Args
    args = create_default_args(
        {
            "cuda": 0,
            "n_inner_updates": 5,
            "second_order": True,
            "grad_clip_norm": 1.0,
            "learn_lr": True,
            "lr_alpha": 0.25,
            "sync_update": False,
            "mem_size": 200,
            "lr": 0.1,
            "train_mb_size": 10,
            "train_epochs": 10,
            "seed": None,
            "optimizer": "sgd",
            "eps": 1e-6,
        },
        config,
    )

    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )
    # Benchmark
    benchmark = avl.benchmarks.SplitCIFAR100(n_experiences=20, return_task_id=True)

    # Loggers and metrics
    interactive_logger = avl.logging.InteractiveLogger()
    tb_logger = avl.logging.TensorboardLogger(tb_log_dir=f"tb")

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, tb_logger],
    )

    # Buffer
    rs_buffer = ReservoirSamplingBuffer(max_size=args.mem_size)
    replay_plugin = ReplayPlugin(
        mem_size=args.mem_size,
        batch_size=args.train_mb_size,
        batch_size_mem=args.train_mb_size,
        task_balanced_dataloader=False,
        storage_policy=rs_buffer,
    )

    # Strategy
    model = MTConvCIFAR()

    optimizer = get_optimizer(
        model.parameters(),
        optimizer=args.optimizer,
        lr=args.lr,
        eps=args.eps,
    )

    cl_strategy = LaMAML(
        model,
        torch.optim.SGD(model.parameters(), lr=args.lr),
        CrossEntropyLoss(),
        n_inner_updates=args.n_inner_updates,
        second_order=args.second_order,
        grad_clip_norm=args.grad_clip_norm,
        learn_lr=args.learn_lr,
        lr_alpha=args.lr_alpha,
        sync_update=args.sync_update,
        train_mb_size=args.train_mb_size,
        train_epochs=args.train_epochs,
        eval_mb_size=100,
        device=device,
        plugins=[replay_plugin],
        evaluator=evaluation_plugin,
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)
    return res


if __name__ == "__main__":
    config_2ord = {
        "lr": tune.grid_search([1, 2.5e-1, 1e-1, 2.5e-2, 1e-2]),
        "optimizer": tune.grid_search(["shampoo", "krad", "kradmm"]),
        "seed": tune.grid_search([100, 200, 300]),
        "eps": tune.grid_search([1e-4]),
    }
    config_1ord = {
        "lr": tune.grid_search([1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]),
        "optimizer": tune.grid_search(["sgd", "adam"]),
        "seed": tune.grid_search([100, 200, 300]),
        "eps": tune.grid_search([1e-6]),
    }

    trainer = tune.with_resources(
        tune.with_parameters(lamaml_scifar100), resources={"cpu": 1, "gpu": 1.0 / 12.0}
    )

    tune_config = tune.TuneConfig(
        metric="Top1_Acc_Stream/eval_phase/test_stream/Task019",
        mode="max",
        num_samples=1,
    )
    tuner = tune.Tuner(trainer, param_space=config_1ord, tune_config=tune_config)
    results_1ord = tuner.fit()

    tuner = tune.Tuner(trainer, param_space=config_2ord, tune_config=tune_config)
    results_2ord = tuner.fit()
