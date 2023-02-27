import torch

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
    double=False,
    beta2=1,
    iterative_roots=False,
):
    hps = kradagrad.HyperParams(
        matrix_eps=eps,
        weight_decay=weight_decay,
        graft_type=0,
        beta2=beta2,
        block_size=block_size,
        best_effort_shape_interpretation=True,
        inverse_exponent_override=inverse_exponent_override,
        preconditioning_compute_steps=precon_update_freq,
        double=double,
        iterative_matrix_roots=iterative_roots,
    )
    if optimizer == "shampoo":
        optimizer = kradagrad.Shampoo(params, lr=lr, hyperparams=hps, momentum=momentum)
    elif optimizer == "krad":
        optimizer = kradagrad.KradagradMM(
            params,
            lr=lr,
            hyperparams=hps,
            momentum=momentum,
            debug=debug,
        )
    elif optimizer == "krad*":
        optimizer = kradagrad.KradagradPP(
            params,
            lr=lr,
            momentum=momentum,
            hyperparams=hps,
            tensor_batch_size=0,
        )  # tensor batching only for single run

    elif optimizer in ["sgd", "learned_lr"]:
        optimizer = torch.optim.SGD(
            params, lr, weight_decay=weight_decay, momentum=momentum
        )
    elif optimizer == "adam":
        optimizer = torch.optim.AdamW(params, lr, weight_decay=weight_decay, eps=eps)

    return optimizer
