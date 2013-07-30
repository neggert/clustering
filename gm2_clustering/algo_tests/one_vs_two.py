import matplotlib.pylab as plt
from gm2_clustering.wf_generator import *
import math
import numpy as np


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

    n_one_tries = 10
    n_one_correct = 0
    for _ in xrange(n_one_tries):
        params, wf = generate_one()
        n = len(fit_func(wf))
        if n == 1:
            n_one_correct += 1

    print("False-positive rate: {}".format(1-1.*n_one_correct/n_one_tries))

    n_two_tries = 10000
    incorrect_dr, total_dr = [], []
    incorrect_dt, total_dt = [], []

    for _ in xrange(n_two_tries):
        p1, p2, wf = generate_two()
        n = len(fit_func(wf))
        dr = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        dt = abs(p1[2]-p2[2])
        if n < 2:
            incorrect_dr.append(dr)
            incorrect_dt.append(dt)
        total_dr.append(dr)
        total_dt.append(dt)

    total_count, bin_x, bin_y = np.histogram2d(total_dr, total_dt, bins=(16, 50))
    incorrect_count, bin_x, bin_y = np.histogram2d(incorrect_dr, incorrect_dt, bins=[bin_x, bin_y])

    extent = [bin_y[0], bin_y[-1], bin_x[0], bin_x[-1]]
    plt.imshow(1.*incorrect_count/total_count, extent=extent, interpolation='nearest', aspect='auto', origin='lower')
    plt.ylabel(r"$\Delta R$ (crystals)")
    plt.xlabel(r"$\Delta t$ (ns)")
    cb = plt.colorbar()
    cb.set_label("False-negative rate")
    plt.show()
    raw_input()