import numpy as np


def fit_waveform(wf, thresh_frac=0.3, count_limit=6):
    """
    Fit a waveform. Decides whether there are 1 or two signals present
    by counting the number of time samples above a threshold. Sums over
    spatial information.

    Params:
    wf - (5*7*?) numpy array giving the digitized waveform
    thresh_frac - fraction of the maximum to use as the threshold

    Returns:
    ((x0, y0, t0), ...) List of tuples. One tuple for each cluster the algorithm find.
                        Each tuple gives the centroid of the cluster in x, y, and time.
    """

    # ignore all spatial information
    ts = wf.sum(axis=(0, 1))

    # find a threhold
    threshold = thresh_frac*ts.max()

    # find the number of samples above the threshold
    counts = np.count_nonzero(ts > threshold)

    if counts > count_limit:
        return ((None, None, None), (None, None, None))
    else:
        return ((None, None, None), )

if __name__ == '__main__':
    import gm2_clustering

    gm2_clustering.algo_tests.one_vs_two(fit_waveform, gm2_clustering.wf_generator.baseline,
                                         gm2_clustering.wf_generator.baseline_two)
