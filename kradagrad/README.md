# KrADagrad

PyTorch implementation of Kronecker Approximation-Domination preconditioned gradient optimization

- Main code lives in in `kradagrad.py`
- We include `shampoo` in `third_party/` without modification for convenience
- Methods for taking p-th roots of symmetric positive semi-definite matrices live in `matrix_root.py`
- An attempt at optimizing performance lives in `kradagrad_batched.py`
- Experiments (notebooks, training scripts) will be found in `experiments/`

