import numpy as np
import scipy.stats


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
