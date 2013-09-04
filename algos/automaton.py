"""
Clustering using cellular automaton
"""

import scipy.signal
import numpy as np
import numpy.ma as ma
from numba import autojit
from copy import deepcopy


@autojit
def get_local_maxima(wf, threshold=200):
    """Find the indices of any local maxima"""
    imax, jmax, kmax = wf.shape
    results = []
    for i in xrange(imax):
        for j in xrange(jmax):
            for k in xrange(kmax):
                x = wf[i, j, k]
                surrounding = wf[max([i-1, 0]):min([i+2, imax]),
                                 max([j-1, 0]):min([j+2, jmax]),
                                 max([k-1, 0]):min([k+2, kmax])]
                if np.max(surrounding) == x and x > threshold:
                    results.append((i, j, k))
    return results


@autojit
def automaton_iteration(tags):
    """
    Do one iteration of the automaton
    """
    imax, jmax, kmax = tags.shape
    start_tags = deepcopy(tags)
    for i in xrange(imax):
        for j in xrange(jmax):
            for k in xrange(kmax):
                tag = tags[i, j, k]
                if tag == -1:
                    continue
                elif len(tag) > 0:
                    continue
                # look for neighboring tags
                neighbor_tags = set()
                for ii in xrange(max([i-1, 0]), min([i+2, imax])):
                    for jj in xrange(max([j-1, 0]), min([j+2, jmax])):
                        for kk in xrange(max([k-1, 0]), min([k+2, kmax])):
                            if start_tags[ii, jj, kk] != -1:
                                neighbor_tags.update(start_tags[ii, jj, kk])
                tag.update(neighbor_tags)
    return tags


def initialize_tags(wf, threshold, seed_threshold=200):
    """
    Get local maxima to start the propagation and set
    entries below the noise floor to -1
    """

    maxima = get_local_maxima(wf, seed_threshold)

    # hack to initialize a numpy array of sets
    lister = np.frompyfunc(lambda x: set(), 1, 1)
    tags = np.zeros(wf.shape, dtype=np.object)
    lister(tags, tags)
    tags[wf < threshold] = -1
    for i, m in enumerate(maxima):
        tags[m].add(i+1)
    return tags


def automaton(wf, threshold):
    """
    Grow calorimeter clusters using a cellular automaton.
    Work in progress
    """

    # generate the starting tags
    tags = initialize_tags(wf, threshold, 200)

    # run the automaton until it converges
    tag_last = np.zeros(wf.shape, dtype=np.object)
    while not np.all(tag_last == tags):
        tag_last[:] = tags
        tags[:] = automaton_iteration(tags)
    return tags


def _has_tag(x, y):
    if x == -1:
        return False
    else:
        return y in x


_has_tag_obj = np.frompyfunc(_has_tag, 2, 1)


def has_tag(tags, itag):
    """Return boolean array indicating whether each tags entry contains itag"""
    return _has_tag_obj(tags, itag).astype(np.bool)


def get_clusters(wf, threshold=1):
    """
    Generator over clusters. Each cluster is the whole waveform array, with only 
    the crystals in that cluster unmasked.
    """
    time_offset, trimmed_wf = make_time_island(wf)
    tags = automaton(wf, threshold)
    return get_tagged(wf, tags)


def get_tagged(wf, tags):
    all_clusters = reduce(set.union, tags[tags != -1].ravel(), set())

    masked_wf = wf.view(ma.MaskedArray)
    masked_wf.fill_value = 0.
    for icluster in all_clusters:
        masked_wf.mask = False  # unmask everything
        masked_wf[~has_tag(tags, icluster)] = ma.masked

        yield ma.copy(masked_wf)


def make_time_island(wf, threshold=50, window=5):
    ts = np.sum(wf, axis=(0, 1))
    count = len(ts[ts > threshold])
    while count < 3:
        threshold /= 2.
        count = len(ts[ts > threshold])
    t_min = np.min(np.arange(ts.shape[0])[ts > threshold]) - window
    t_max = np.max(np.arange(ts.shape[0])[ts > threshold]) + 3*window
    if t_min < 0:
        t_min = 0
    if t_max > len(ts):
        t_max = len(ts)
    return t_min, wf[:, :, t_min:t_max]


def fit_waveform(wf, threshold=1.):
    """
    Fit the waveform. Currently just want to see if there are one or two
    clusters
    """

    time_offset, trimmed_wf = make_time_island(wf)
    clusters = automaton(trimmed_wf, threshold)
    all_clusters = reduce(set.union, clusters[clusters != -1].ravel(), set())
    if len(all_clusters) > 1:
        return ((None, None, None), (None, None, None))
    else:
        return ((None, None, None),)

def test():
    import gm2_clustering
    data = np.load("processed_wf/test_sim.npz")
    p1, p2, wf = data['two'][0]
    c = get_clusters(wf)
    gm2_clustering.plot_waveform(c.next(), c.next())


if __name__ == '__main__':
    import gm2_clustering

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="File in which waveforms are stored")

    args = parser.parse_args()

    gm2_clustering.algo_tests.one_vs_two(fit_waveform, args.data_file)
