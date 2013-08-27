"""
Check for one or two pulses using a likelihood ratio test
"""

import numpy as np
import numpy.ma as ma
import scipy.stats
import scipy.optimize

import algos.automaton as ca


def make_coords():
    """
    Generate the coordinates for the waveform.
    returns a list of numpy arrays (x,y,t), giving
    the x, y, and t positions at each grid point.
    """
    return np.meshgrid(np.arange(0, 7, 1), np.arange(0, 5, 1),
                       np.linspace(0, 100, 50))


def gaussian_beta(x0, y0, t0, amplitude, xwidth, ywidth):
    """
    Generate a waveform using a gaussian for the spatial distribution
    and a beta function for the pulse shape.

    Parameters:
        x0: x position of the center of the shower.
        y0: y position of the center of the shower.
        t0: time of the start of the shower.
        amplitude: height of the shower
        xwidth, ywidth: size of the shower and x and y directions
                        No angled showers for now.
    """

    x, y, t = make_coords()

    wf = scipy.stats.gamma.pdf(t, 1.4, loc=t0, scale=5.)
    wf *= amplitude
    wf *= scipy.stats.norm.pdf(x, loc=x0, scale=xwidth)
    wf *= scipy.stats.norm.pdf(y, loc=y0, scale=ywidth)

    return wf


def chi2_one_pulse(wf, pulse_one_params):
    """Calculate the chi-squared for the one-pulse hypothesis"""

    fitted = gaussian_beta(*pulse_one_params)
    return np.sum((fitted-wf)**2)


def chi2_two_pulse(wf, pulse_one_params, pulse_two_params):
    """Calculate the chi-squared for the one-pulse hypothesis"""

    fitted = gaussian_beta(*pulse_one_params) + gaussian_beta(*pulse_two_params)
    return np.sum((fitted-wf)**2)


def profile_chi2_one_pulse(wf):
    """
    Minimize chi-squared over the free parameters to get the
    "profile" chi2
    """
    def func_to_minimize(args):
        fitted = gaussian_beta(*args)
        return (fitted-wf).ravel()

    # initial guesses
    # just pick the maximum
    yinit, xinit, tinit = np.unravel_index(np.argmax(wf), wf.shape)
    Ainit = np.sum(wf)*3
    xwinit, ywinit = 1, 1

    result, _, info, mesg, ier = scipy.optimize.leastsq(func_to_minimize,
                                                        x0=(xinit, yinit, tinit, Ainit, xwinit, ywinit),
                                                        full_output=1
                                                        )
    return chi2_one_pulse(wf, result)/wf[-wf.mask].shape[0]


def profile_chi2_two_pulse(wf):
    """
    Minimize chi-squared over the free parameters to get the
    "profile" chi2
    """

    n_args = 6

    def func_to_minimize(args):
        fitted = gaussian_beta(*args[:n_args])+gaussian_beta(*args[n_args:])
        return (fitted-wf).ravel()

    # initial guesses
    # just pick the maximum
    yinit, xinit, tinit = np.unravel_index(np.argmax(wf), wf.shape)
    Ainit, xwinit, ywinit = 1000, 1, 1

    # mask the region around that maximum to get the second waveform guesses
    m = wf.view(ma.MaskedArray)

    def or_zero(x):
        if x < 0:
            return 0
        return x

    m[or_zero(yinit-1):(yinit+2), or_zero(xinit-1):(xinit+2), or_zero(tinit-4):(tinit+5)] = ma.masked
    y2init, x2init, t2init = np.unravel_index(np.argmax(m), m.shape)
    A2init, xw2init, yw2init = 1000, 1, 1

    result, _, info, mesg, ier = scipy.optimize.leastsq(func_to_minimize,
                                                        x0=(xinit, yinit, tinit, Ainit, xwinit, ywinit,
                                                            x2init, y2init, t2init, A2init, xw2init, yw2init),
                                                        full_output=1,
                                                        maxfev = 50000
                                                        )
    if ier not in range(1, 5):
        print mesg
    # print ier, chi2_two_pulse(wf, result[:n_args], result[n_args:])

    return chi2_two_pulse(wf, result[:n_args], result[n_args:])

def get_llr(wf):
    return profile_chi2_one_pulse(wf)-profile_chi2_two_pulse(wf)


def get_one_pulse_llr_dist(filename, n):
    llrs = []
    with np.load(filename) as data:
        for datum in data['one'][:n]:
            params, wf = datum
            cluster = ca.get_clusters(wf, 1.).next()
            llr = get_llr(cluster)
            llrs.append(llr)

    return llrs

def get_two_pulse_llr_dist(filename, n):
    llrs = []
    with np.load(filename) as data:
        for datum in data['two'][:n]:
            _,_, wf = datum
            llr = get_llr(wf)
            llrs.append(llr)

    return llrs

if __name__ == '__main__':
    get_one_pulse_llr_dist("test.npz", 100)
