import scipy.stats


def middle(**kwargs):
    return 2.5, 3.5, 10


def uniform(**kwargs):
    x = scipy.stats.uniform.rvs(loc=0, scale=7)
    y = scipy.stats.uniform.rvs(loc=0, scale=5)
    t = scipy.stats.uniform.rvs(loc=25, scale=50)

    return x, y, t
