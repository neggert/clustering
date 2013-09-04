import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_colors = ['b', 'r', 'y']

def plot_waveform(*args):
    """Each item in args should be a waveform"""
    plt.figure(figsize=(30, 24))
    gs = gridspec.GridSpec(6, 9)
    y_up = 0.
    x_up = 0.
    for data in args:
        y_up = max([data.max(), y_up])
        x_up = max([data.shape[2], x_up])
    y_up *= 1.1
    for i in xrange(9):
        for j in xrange(6):
            ax = plt.subplot(gs[j, i])
            colc = iter(_colors)
            for data in args:
                y = data[i, j, :]
                col = colc.next()
                ax.scatter(range(len(y)), y, color=col, edgecolors='k', s=30, alpha=0.5)
            ax.set_ylim(0.1, y_up)
            ax.set_xlim(0, x_up)
            ax.set_yscale("log", nonposy="clip")
    plt.tight_layout()
