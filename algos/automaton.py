"""
Clustering using cellular automaton
"""

import scipy.signal
import numpy as np
from numba import autojit

@autojit
def get_local_maxima(wf):
    """Find the indices of any local maxima"""
    imax, jmax, kmax = wf.shape
    results = []
    for i in xrange(imax):
        for j in xrange(jmax):
            for k in xrange(kmax):
                x = wf[i,j,k]
                surrounding = wf[max([i-1, 0]):min([i+2, imax]),
                                 max([j-1, 0]):min([j+2, jmax]),
                                 max([k-1, 0]):min([k+2, kmax])]
                if np.max(surrounding) == x and x > 1:
                    results.append((i,j,k)) 
    return results


@autojit
def automaton_iteration(tags):
    """
    Do one iteration of the automaton
    """
    imax, jmax, kmax = tags.shape
    for i in xrange(imax):
        for j in xrange(jmax):
            for k in xrange(kmax):
                tag = tags[i, j, k]
                if tag == -1:  # yes, this doesn't follow P
                    continue
                surrounding = tags[max([i-1, 0]):min([i+2, imax]),
                                   max([j-1, 0]):min([j+2, jmax]),
                                   max([k-1, 0]):min([k+2, kmax])]
                # look for neighboring tags
                neighbor_tags = reduce(set.union, surrounding[surrounding != -1].ravel(), set())
                tag.update(neighbor_tags)

    return tags


def initialize_tags(wf, threshold):
    maxima = get_local_maxima(wf)

    # hack to initialize a numpy array of sets
    lister = np.frompyfunc(lambda x: set(), 1, 1)
    tags = np.zeros(wf.shape, dtype=np.object)
    lister(tags, tags)
    tags[wf < threshold] = -1
    for i, m in enumerate(maxima):
        tags[m].add(i+1)
    return tags


def automaton(wf, threshold):

    # generate the starting tags
    tags = initialize_tags(wf, threshold)

    # run the automaton until it converges
    tag_last = np.zeros(wf.shape, dtype=np.object)
    while not np.all(tag_last == tags):
        tag_last[:] = tags
        tags[:] = automaton_iteration(tags)
    return tags

def fit_waveform(wf, threshold=1.):
    clusters = automaton(wf, threshold)
    all_clusters = reduce(set.union, clusters[clusters != -1].ravel(), set())
    if len(all_clusters) > 2:
        return ((None, None, None), (None, None, None))
    else:
        return ((None, None, None),)

if __name__ == '__main__':
    import gm2_clustering

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="File in which waveforms are stored")

    args = parser.parse_args()

    gm2_clustering.algo_tests.one_vs_two(fit_waveform, args.data_file)