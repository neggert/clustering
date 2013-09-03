import numpy as np
import scipy.ndimage.interpolation

def load_waveform_file(filename):
    """Load the waveforms from a binary file. Returns an array of waveforms, each with 6*9*200 entries"""
    wfs = np.fromfile(filename, np.uint16)
    split_wfs = np.split(wfs, len(wfs)/6/9/200)
    return split_wfs


def make_coords():
    """
    Generate the coordinates for the waveform.
    returns a list of numpy arrays (x,y,t), giving
    the x, y, and t positions at each grid point.
    """
    return np.meshgrid(np.arange(0, 9, 1), np.arange(0, 6, 1),
                       np.linspace(0, 398, 200), indexing='ij')


def interpolate_waveform(wf, x, y, t):  # Could vectorize this easily
    """
    Evaluate the waveform at arbitary x,y,t through interpolation.

    Inputs:
    wf      The waveform. 9x6x200 numpy array
    x,y,t   Position to evaluate the waveform function at

    Output:
    value   The interpolated value
    """

    # need to do unit conversions on this line
    # 1 sample = 2 ns
    # x, y currently measured in crystals
    return scipy.ndimage.interpolation.map_coordinates(wf, np.vstack((x,y,t*2)), order=1, mode="constant")
