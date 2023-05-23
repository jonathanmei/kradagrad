def roundup(p, m):
    # how many groups of m are in p?
    # p: int or float
    # m: int
    return int(p // m) + int(p % m > 0)