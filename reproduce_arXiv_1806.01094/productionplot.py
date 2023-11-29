import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('pgf', force=True)

# latex font sizes at base 11pt
# tiny 6
# scriptsize 8
# footnotesize 9
# small 10
# normalsize 11
# large 12
# Large 14
# LARGE 17
# huge 20
# Huge 25

matplotlib.rcParams.update({
    # font sizes
    'font.size': 10,
    'axes.titlesize': 9,
    'legend.fontsize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    # activate tex rendering
    'text.usetex': True,
    # tex settings
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': "\n".join([
        r'\usepackage[utf8]{inputenc}',
        r'\usepackage{dsfont}',
        r'\usepackage{amsmath,amssymb,amsfonts,mathrsfs}'
    ])
})

plt.style.use('seaborn-dark')

TEXTWIDTH = 6.00117
TEXTHEIGHT = 8.50166


def colormapping(methods):
    methods = [m[0] for m in methods]
    colors = [None] * len(methods)
    no_coroICA = len([m for m in methods if 'coro' in m])
    no_uwedgeICA = len([m for m in methods if 'uwedge' in m])
    counter_coroICA = 0
    counter_uwedgeICA = 0
    for k, method in enumerate(methods):
        if method == 'random':
            colors[k] = matplotlib.colors.to_hex(matplotlib.cm.get_cmap('RdBu')(
                0.5 + 3 / 12)),
        elif method == 'fastICA':
            colors[k] = 'darkblue'
        elif 'coro' in method:
            colors[k] = matplotlib.colors.to_hex(matplotlib.cm.get_cmap('PRGn')(
                0.5 + (counter_coroICA + 1) / (no_coroICA) / 2)),
            counter_coroICA += 1
        elif 'uwedge' in method:
            colors[k] = matplotlib.colors.to_hex(matplotlib.cm.get_cmap('RdBu')(
                (counter_uwedgeICA) / (no_uwedgeICA + 1) / 2)),
            counter_uwedgeICA += 1
        else:
            colors[k] = '#979797'
    return colors
