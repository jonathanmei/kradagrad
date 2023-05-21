import os
import sys

import avalanche as avl
import kradagrad.math_utils as mu
import ray
import torch
from avalanche.benchmarks.utils import AvalancheSubset
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.evaluation import metrics as metrics
from experiments.utils import create_default_args, set_seed
from kiwisolver import Expression
from models import MLP
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
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


def train_gem_pmnist(config):
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
            "train_mb_size": 10,
            "seed": None,
        }
    )

    run_name = f"{config['optimizer']}_lr_{config['lr']}_seed_{config['seed']}"

    set_seed(config["seed"])
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
    tb_logger = avl.logging.TensorboardLogger(tb_log_dir=f"tb")

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, tb_logger],
    )

    optimizer = get_optimizer(
        model.parameters(),
        optimizer=config["optimizer"],
        lr=config["lr"],
        eps=config["eps"],
    )

    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

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

    os.makedirs("my_model", exist_ok=True)
    torch.save((model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
    checkpoint = Checkpoint.from_directory("my_model")

    session.report(res, checkpoint=checkpoint)

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

    tune_config = tune.TuneConfig(
        metric="Top1_Acc_Stream/eval_phase/test_stream/Task000",
        mode="max",
        num_samples=1,
    )
    # tuner = tune.Tuner(
    #     train_gem_pmnist, param_space=config_1ord, tune_config=tune_config
    # )

    # results_1ord = tuner.fit()

    tuner = tune.Tuner(
        train_gem_pmnist, param_space=config_2ord, tune_config=tune_config
    )
    # tuner = tuner.restore("~/ray_results/train_gem_pmnist_2023-01-10_03-58-50")
    results_2ord = tuner.fit()
