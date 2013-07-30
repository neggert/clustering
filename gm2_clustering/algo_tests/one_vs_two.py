import matplotlib.pylab as plt
from gm2_clustering.wf_generator import *


def one_vs_two(fit_func, generate_one, generate_two):
    """
    Test performance of an algorithm at distinguishing one-electron
    events from two-electron events.

    Parameters:
    fit_func - function that takes a single input, the waveform, and outputs a tuple
               The length of the tuple is the number of electrons the algorithm thinks
               were in the waveform.
    generate_one - function to generate a waveform from one electron
    generate_two - function to generate a waveform from two electrons

    Output:
    There are two relevant measures here.

        * The fake rate: How often the algorithm thinks there were two electrons when
        there was really only one.
        * The efficiency: How often the algorithm correctly identifies two-electron
        waveforms as a function of the distance between the two waveforms in space
        and time.
    """

    n_one_tries = 1000
    n_one_correct = 0
    for _ in xrange(n_one_tries):
        params, wf = generate_one()
        n = fit_func(wf)
        if n == 1:
            n_one_correct += 1

    print("Two-electron mis-tag rate: {}".format([1.*n_one_correct/n_one_tries]))
