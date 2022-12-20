from itertools import chain, product
import os
import subprocess
import sys
from types import SimpleNamespace

def gen_arg(dict_add):
    args_base = SimpleNamespace()
    for k_, v_ in dict_add.items():
        args_base.__setattr__(k_, v_)
    return args_base

def gen_sweep(dict_sweep):
    # dict of lists
    for prod_ in product(*list(dict_sweep.values())):
        yield gen_arg(dict(zip(dict_sweep.keys(), prod_)))
        
def run_sweep(args_sweep, n_proc, n_gpu, timeout):
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
                 'python', 'cifar10.py',
                 '--optimizer', args.optimizer,
                 '--epochs', args.epochs,
                 '--lr_str', args.lr_str,
                 '--eps_str', args.eps_str
                ]),
                shell=True
            ))
            # really only used at the start to fill `running_procs`
            last_gpu = (last_gpu + 1) % n_gpu


hyperparams_to_sweep_precon = SimpleNamespace(
    optimizer=['kradmm', 'krad', 'shampoo'],  
    epochs=['250'],
    eps_str=['1e-6', '1e-4', '1e-2'],
    lr_str=['1', '2.5e-1', '1e-1', '2.5e-2', '1e-2'],
)
args_sweep_precon = gen_sweep(hyperparams_to_sweep_precon.__dict__)

hyperparams_to_sweep_gd = SimpleNamespace(
    optimizer=['sgd'],
    epochs=['250'],
    eps_str=['1e-1'],
    lr_str=['1e-1', '5e-2', '2e-2', '1e-2', '5e-3', '2e-3', '1e-3'],
) 
args_sweep_gd = gen_sweep(hyperparams_to_sweep_gd.__dict__)

hyperparams_to_sweep_ada = SimpleNamespace(
    optimizer=['ada'],
    epochs=['250'],
    eps_str=['1e-1'],
    lr_str=['1e-1', '5e-2', '2e-2', '1e-2', '5e-3', '2e-3', '1e-3'],
)
args_sweep_ada = gen_sweep(hyperparams_to_sweep_ada.__dict__)
n_proc = 24
n_gpu = 2
timeout = 10
args_sweeps = chain(args_sweep_precon, args_sweep_gd, args_sweep_ada)
run_sweep(args_sweeps, n_proc, n_gpu, timeout)