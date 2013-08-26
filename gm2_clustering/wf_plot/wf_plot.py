import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_waveform(data):
    plt.figure(figsize=(24, 16))
    up = data.max()
    gs = gridspec.GridSpec(6, 9)
    for i in xrange(6):
        for j in xrange(9):
            x = data[i, j, :]
            ax = plt.subplot(gs[i, j])
            ax.plot(x)
            ax.set_ylim(0.1, up)
            ax.set_yscale("log", nonposy="clip")
    plt.tight_layout()
