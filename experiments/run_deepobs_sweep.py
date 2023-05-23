from itertools import chain, product
import os
import subprocess
import sys
from types import SimpleNamespace as Sn

def gen_arg(dict_add):
    args_base = Sn()
    for k_, v_ in dict_add.items():
        args_base.__setattr__(k_, v_)
    return args_base

def gen_sweep(dict_sweep):
    # dict of lists
    for prod_ in product(*list(dict_sweep.values())):
        yield gen_arg(dict(zip(dict_sweep.keys(), prod_)))

def run_sweep(args_sweep, n_proc, n_gpu, timeout, hundred=False):
    # poll every `timeout` seconds
    running_procs = []
    last_gpu = 0
    for args in args_sweep:
        while len(running_procs) == n_proc:
            for proc in running_procs:
                try:
                    proc.wait(timeout)
                    last_gpu = int(proc.args.split(' ')[0].split('=')[-1])
                    running_procs.remove(proc)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if len(running_procs) < n_proc:
            running_procs.append(subprocess.Popen(' '.join(
                ['CUDA_VISIBLE_DEVICES={}'.format(last_gpu),
                 'python', 'deepobs_single.py',
                 args.prob,
                 #'cifar10_3c3d',
                 args.optimizer,
                 '--lr', args.lr_str,
                 '--epochs', '350',
                 '--preconditioning_compute_steps', '50',
                 '>', 'results/sweep_{}_{}_{}.out'.format(args.prob, args.optimizer, args.lr_str), '2>&1'
                 ]),
                shell=True
            ))
            # really only used at the start to fill `running_procs`
            last_gpu = (last_gpu + 1) % n_gpu


if __name__ == '__main__':
    #probs_to_run = ['cifar10_3c3d', 'cifar100_allcnnc', 'fmnist_mlp', 'mnist_vae']
    probs_to_run = ['cifar10_3c3d', 'cifar100_allcnnc', 'tolstoi_char_rnn']

    hyperparams_to_sweep_precon = Sn(
        prob=probs_to_run,
        optimizer=['kradmm'],  
        lr_str=['1e-1', '2.5e-2', '1e-2', '2.5e-3', '1e-3', '2.5e-4'],
    )
    args_sweep_precon_0 = gen_sweep(vars(hyperparams_to_sweep_precon))

    hyperparams_to_sweep_precon = Sn(
        prob=probs_to_run,
        optimizer=['krad', 'shampoo'],  
        lr_str=['1e-1', '2.5e-2', '1e-2', '2.5e-3', '1e-3'],
    )
    args_sweep_precon_1 = gen_sweep(vars(hyperparams_to_sweep_precon))

    # hyperparams_to_sweep_gd = Sn(
    #     prob=probs_to_run,
    #     optimizer=['sgd'],
    #     lr_str=['1e-1', '5e-2', '2e-2', '1e-2', '5e-3', '2e-3', '1e-3'],
    # ) 
    # args_sweep_gd = gen_sweep(vars(hyperparams_to_sweep_gd))

    # hyperparams_to_sweep_ada = Sn(
    #     prob=probs_to_run,
    #     optimizer=['adam'],
    #     lr_str=['5e-2', '2e-2', '1e-2', '5e-3', '2e-3', '1e-3', '5e-4'],
    # )
    # args_sweep_ada = gen_sweep(vars(hyperparams_to_sweep_ada))
    n_proc = 24
    n_gpu = 2
    timeout = 60
    #args_sweeps = chain(args_sweep_gd, args_sweep_ada, args_sweep_precon)
    args_sweeps = chain(args_sweep_precon_0, args_sweep_precon_1)
    run_sweep(args_sweeps, n_proc, n_gpu, timeout)
