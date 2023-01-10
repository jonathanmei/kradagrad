import os
import sys

import avalanche as avl
import kradagrad.math_utils as mu
import torch
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.evaluation import metrics as metrics
from experiments.utils import create_default_args, set_seed
from models import MLP
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

sys.path.append(os.path.expanduser("~/code/experiments/"))

import kradagrad


class GEM_reduced(avl.training.GEM):
    def make_train_dataloader(
        self, num_workers=0, shuffle=True, pin_memory=True, **kwargs
    ):
        """Select only 1000 patterns for each experience as in GEM paper."""
        self.dataloader = TaskBalancedDataLoader(
            AvalancheSubset(self.adapted_dataset, indices=list(range(1000))),
            oversample_small_groups=True,
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )


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


def gem_pmnist(override_args=None):
    """
    "Gradient Episodic Memory for Continual Learning" by Lopez-paz et. al. (2017).
    https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
    """
    args = create_default_args(
        {
            "cuda": 0,
            "patterns_per_exp": 1000,
            "hidden_size": 100,
            "hidden_layers": 2,
            "epochs": 1,
            "dropout": 0,
            "mem_strength": 0.5,
            "learning_rate": 0.1,
            "train_mb_size": 10,
            "seed": None,
            "optimizer": "sgd",
            "lr": 0.01,
            "eps": 1e-6,
        },
        override_args,
    )

    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    benchmark = avl.benchmarks.PermutedMNIST(20)
    model = MLP(
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        drop_rate=args.dropout,
    )
    criterion = CrossEntropyLoss()

    interactive_logger = avl.logging.InteractiveLogger()
    tb_logger = avl.logging.TensorboardLogger(
        tb_log_dir="continual_learning_tflogs/shampoo_20"
    )

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, tb_logger],
    )

    optimizer = get_optimizer(
        model.parameters(), optimizer=args.optimizer, eps=args.eps, lr=args.lr
    )
    cl_strategy = GEM_reduced(
        model,
        optimizer,
        criterion,
        patterns_per_exp=args.patterns_per_exp,
        memory_strength=args.mem_strength,
        train_mb_size=args.train_mb_size,
        train_epochs=args.epochs,
        eval_mb_size=128,
        device=device,
        evaluator=evaluation_plugin,
    )

    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res


if __name__ == "__main__":
    res = gem_pmnist(
        override_args={"optimizer": "kradmm", "lr": 0.01, "seed": 100, "eps": 1e-4}
    )
    print(res)
