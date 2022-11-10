import torch

# Simplified version of 
# https://github.com/shril/CubicEquationSolver
# Algorithm Link: www.1728.org/cubic2.htm

def solve_smallest(a, b, c, d, thr=-float('inf')):
    # Takes in batches of coefficients of the Cubic Polynomial
    # as parameters and returns the largest root in a torch.Tensor.
    # ax^3 + bx^2 + cx + d = 0
    # Assumes we actually have cubic (i.e. a != 0)

    # input shapes: (n, )
    # output shape: (n, )

    # TODO: vectorize
    f = findF(a, b, c)
    g = findG(a, b, c, d)
    h = findH(g, f)

    # trip_root = (f == 0) && (g == 0) && (h == 0)            # All 3 Roots are Real and Equal
    x_trip = ((d / a) ** (1 / 3.0) * -1)
    x_trip[d / a < 0] = ((-d / a) ** (1 / 3.0))[d / a < 0]

    all_real = h <= 0                                # All 3 roots are Real

    i = torch.sqrt(((g ** 2.0) / 4.0) - h)
    j = i ** (1 / 3.0)
    k = torch.acos(-(g / (2 * i)))
    L = j * -1
    M = torch.cos(k / 3.0)
    N = torch.sqrt(torch.Tensor([3])) * torch.sin(k / 3.0)
    P = (b / (3.0 * a)) * -1

    x1 = 2 * j * torch.cos(k / 3.0) - (b / (3.0 * a))
    x2 = L * (M + N) + P
    x3 = L * (M - N) + P
    x1[x1<thr] = float('inf')
    x2[x2<thr] = float('inf')
    x3[x3<thr] = float('inf')
    # like torch.minimum, but ignores nan's:
    x_real = torch.fmin(torch.fmin(x1, x2), x3)

    sing_root = h > 0                                 # One Real Root and two Complex Roots
    R = -(g / 2.0) + torch.sqrt(h)
    S = R ** (1 / 3.0)
    S[R < 0] = ((-R) ** (1 / 3.0) * -1)[R < 0]
    T = -(g / 2.0) - torch.sqrt(h)
    U = (T ** (1 / 3.0))
    U[T < 0] = (((-T) ** (1 / 3.0)) * -1)[T < 0]

    x_sing = (S + U) - (b / (3.0 * a))

    x_out = x_trip
    x_out[all_real] = x_real[all_real]
    x_out[sing_root] = x_sing[sing_root]
    return x_out

# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)
