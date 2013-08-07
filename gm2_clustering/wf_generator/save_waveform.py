import argparse
import inspect
import numpy as np
import gm2_clustering.wf_generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="Number of waveforms to generate", type=int)
    parser.add_argument("output_file", help="Output file")
    parser.add_argument("param_fcn", help="Function to generate waveform parameters. Must be a function in gm2_clustering.wf_generator.params")
    parser.add_argument("waveform_fcn", help="Function to generate waveforms. Must be a function in gm2_clustering.wf_generator.waveforms")
    parser.add_argument("transform_fcn", help="Function to transform waveforms. Must be a function in gm2_clustering.wf_generator.transform")

    args, extras = parser.parse_known_args()

    # get the actual functions
    try:
        param_fcn = getattr(gm2_clustering.wf_generator.params, args.param_fcn)
    except AttributeError:
        raise ValueError("{} is not a valid param_fcn".format(args.param_fcn))
    try:
        waveform_fcn = getattr(gm2_clustering.wf_generator.waveforms, args.waveform_fcn)
    except AttributeError:
        raise ValueError("{} is not a valid waveform_fcn".format(args.waveform_fcn))
    try:
        transform_fcn = getattr(gm2_clustering.wf_generator.transform, args.transform_fcn)
    except AttributeError:
        raise ValueError("{} is not a valid transform_fcn".format(args.waveform_fcn))

    # now add the kwargs of those functions as additional arguments

    param_kwargs = inspect.getargspec(param_fcn)[0]
    for kw in param_kwargs:
        if kw is not None:
            parser.add_argument("--{}".format(kw), help="{} argument".format(args.param_fcn), type=float, required=True)
    waveform_kwargs = inspect.getargspec(waveform_fcn)[0]
    for kw in waveform_kwargs[3:]:
        if kw is not None:
            parser.add_argument("--{}".format(kw), help="{} argument".format(args.waveform_fcn), type=float, required=True)
    transform_kwargs = inspect.getargspec(transform_fcn)[0]
    for kw in transform_kwargs[1:]:
        if kw is not None:
            parser.add_argument("--{}".format(kw), help="{} argument".format(args.transform_fcn), type=float, required=True)

    arg2 = parser.parse_args()

    kwargs = {k: v for (k, v) in vars(arg2).iteritems() if k not in vars(args).keys()}

    return args.n, args.output_file, param_fcn, waveform_fcn, transform_fcn, kwargs


def save_waveform(n, output_file, param_fcn, wf_fcn, trans_fcn, kwargs):
    one_iter = gm2_clustering.wf_generator.generate(param_fcn, wf_fcn, trans_fcn, **kwargs)
    two_iter = gm2_clustering.wf_generator.generate_two(param_fcn, wf_fcn, trans_fcn, **kwargs)

    one_data = []
    two_data = []

    for _ in xrange(n):
        one_data.append(one_iter.next())
        two_data.append(two_iter.next())

    np.savez(output_file, one=one_data, two=two_data)

if __name__ == '__main__':
    args = parse_args()
    save_waveform(*args)

