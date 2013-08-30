import numpy as np
import scipy.stats
import itertools
import gm2_clustering.utils as utils


class GaussianBetaWaveform(object):
    def __init__(self):
        self.x, self.y, self.t = make_coords()

    def __call__(self, x0, y0, t0, amplitude, xwidth, ywidth):
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
        wf = scipy.stats.gamma.pdf(self.t, 1.4, loc=t0, scale=5.)
        wf *= amplitude
        wf *= scipy.stats.norm.pdf(self.x, loc=x0, scale=xwidth)
        wf *= scipy.stats.norm.pdf(self.y, loc=y0, scale=ywidth)

        return wf


def transform_wf(wf, x0, y0, t0, amplitude):
    """Transform a waveform to the given coordinates. Currently only amplitude implemented"""
    new_x, new_y, new_t = utils.make_coords()
    new_x -= x0
    new_y -= y0
    new_t -= t0
    new_wf = utils.interpolate_waveform(wf, new_x.ravel(), new_y.ravel(), new_t.ravel())


    return new_wf.reshape(9,6,200)*amplitude


class SimulatedWaveform(object):
    """Load waveforms from a file"""
    def __init__(self, wf_filename):
        self.wfs = itertools.cycle(utils.load_waveform_file(wf_filename))

    def __call__(self, x0, y0, t0, amplitude):
        """
        Return simulated waveforms

        Pulls waveform library from a file. The return values are transformations of these waveforms.
        When it exhausts the waveform library, it starts over from the beginning
        """
        wf = self.wfs.next().reshape(6,9,200).swapaxes(0,1)
        return transform_wf(wf, x0, y0, t0, amplitude)

base_wf = SimulatedWaveform("/home/nic/gm2/clustering/waveforms/waveforms.bin")

def base_sim_wf(x0, y0, t0, amplitude):
    return base_wf(x0, y0, t0, amplitude)