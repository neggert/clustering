import scipy.stats


def middle():
    return 2.5, 3.5, 10


def uniform():
    x = scipy.stats.uniform.rvs(loc=-5, scale=7)
    y = scipy.stats.uniform.rvs(loc=-2, scale=5)
    t = scipy.stats.uniform.rvs(loc=25, scale=150)

    return x, y, t

def uniform_ints():
    x = scipy.stats.randint.rvs(loc=0, scale=9)
    y = scipy.stats.randint.rvs(loc=0, scale=6)
    t = scipy.stats.randint.rvs(loc=0, scale=150)

    return x, y, t