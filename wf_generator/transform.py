import scipy.stats


def gaussian_noise(waveform, noise):
    """
    Add gaussian electronics noise to a waveform

    Parameters:

        waveform: the input waveform to be transformed. Should be
                  a 3-diminsional numpy array

        noise: the amplitude of noise to add

    Returns:

        transformed waveform
    """

    noise = scipy.stats.norm.rvs(scale=noise, size=waveform.shape)
    return waveform+noise
