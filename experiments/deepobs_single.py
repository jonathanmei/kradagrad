import argparse
import os
import sys
from types import SimpleNamespace as Sn

from torch.optim import SGD, Adam
from deepobs import pytorch as pt

parent = os.path.abspath(os.path.join(os.getcwd(), '../..'))
if parent not in sys.path:
    sys.path = [parent] + sys.path
from kradagrad import KradagradPP, KradagradMM, Shampoo

# 2 level SimpleNamespace recursive dict
def dictify(hps):
    return {k_: vars(v_) for k_, v_ in vars(hps).items()}

if __name__ == '__main__':
    opts = Sn(
        krad=KradagradPP,
        kradmm=KradagradMM,
        shampoo=Shampoo,
        adam=Adam,
        sgd=SGD
    )
    prob_choices = ['fmnist_mlp', 'cifar10_3c3d', 'cifar100_allcnnc', 'mnist_vae', 'tolstoi_char_rnn']
    opt_choices = list(vars(opts).keys())

    parser = argparse.ArgumentParser()
    parser.add_argument('prob', type=str, choices=prob_choices, help='which problem to test')
    parser.add_argument('opt', type=str, choices=opt_choices, help='which optimizer to use')
    parser.add_argument('--epochs', type=int, default=50)
    args, remaining_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining_args

    optimizer_class = vars(opts).get(args.opt)
    if args.opt in ['krad', 'kradmm', 'shampoo']:
        hyperparams = Sn(
            lr=Sn(type=float),
            preconditioning_compute_steps=Sn(type=int, default=20),
            matrix_eps=Sn(type=float, default=1e-4),
        )
    elif args.opt in ['adam', 'sgd']:
        hyperparams = Sn(lr=Sn(type=float))
    else:
        raise ValueError('Unknown optimizer {}'.format(args.opt))


    runner = pt.runners.StandardRunner(optimizer_class, dictify(hyperparams))
    runner.run(testproblem=args.prob, num_epochs=args.epochs)

