# KrADagrad

Official PyTorch implementation of 
[_Mei, Jonathan, Alexander Moreno and Luke Walters. "KrADagrad: Kronecker Approximation-Domination Gradient Preconditioned Stochastic Optimization." Uncertainty in Artificial Intelligence (2023)._](https://arxiv.org/abs/2305.19416)


- Main code lives in `kradagrad.py` (KrADagrad* in the paper) and `kradagradmm.py` (KrADagrad in the paper)
- We include `shampoo` in `third_party/` with minor modifications for convenience
- An implementation of resnet also lives in `third_party/` for convenience
- Methods for taking p-th roots (and other useful utilities) for symmetric positive semi-definite matrices live in `positive_matrix_functions.py`
- Experiments (training notebooks+scripts) are found in `experiments/`

Set up environment using conda:
```
conda env create -f krad.yml
```
