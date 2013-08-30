import params
import waveforms
import transform
from generator import generate, generate_two
from functools import partial


# baseline = partial(generate, params.uniform, waveforms.gaussian_beta,
#                    transform.gaussian_noise, amplitude=1000, xwidth=1,
#                    ywidth=1, noise=0.1)

# baseline_two = partial(generate_two, params.uniform, waveforms.gaussian_beta,
#                        transform.gaussian_noise, amplitude=1000, xwidth=1,
#                        ywidth=1, noise=0.1)