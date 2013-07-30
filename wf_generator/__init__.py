import params
import waveforms
import transform
from generator import generate, generate_two
from functools import partial


baseline = partial(generate, params.uniform, waveforms.gaussian_beta,
                   transform.gaussian_noise, amplitude=100, xwidth=1,
                   ywidth=1, noise=3)
