import inspect


def generate(param_fcn, waveform_fcn, transform_fcn, **kwargs):
    """
    Generate a waveform

    Parameters:
        param_fcn: Function that returns x0,y0,t0 for a pulse

        waveform_fcn: Function that accepts (x0, y0, t0, amplitude, **kwargs)
                      and returns a waveform in the form of a 3-d numpy array

        transform_fcn: Function that accepts (waveform, **kwargs) and returns
                       a transformed waveform

    Returns:
        A waveform in the form of a 3-d numpy array giving signal amplitudes
        as a function of x,y, and t. x and y are given in crystal coordinates,
        so 0 < x < 7 and 0 < y < 5. t gives a sample every 2 ns over 100 ns, so
        there are 50 different t values per waveform.
    """

    param_kwargs = {name: val for name, val in kwargs.iteritems()
                    if name in inspect.getargspec(param_fcn)[0]}

    wf_kwargs = {name: val for name, val in kwargs.iteritems()
                 if name in inspect.getargspec(waveform_fcn)[0]}

    transform_kwargs = {name: val for name, val in kwargs.iteritems()
                        if name in inspect.getargspec(transform_fcn)[0]}

    x0, y0, t0 = param_fcn(**param_kwargs)

    wf = waveform_fcn(x0, y0, t0, **wf_kwargs)

    wf = transform_fcn(wf, **transform_kwargs)

    return wf
