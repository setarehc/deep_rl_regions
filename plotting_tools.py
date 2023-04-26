import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

### This part enables plotting pdf files ###
plt.switch_backend("Agg")

mpl.rcParams['lines.markersize'] = 3
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
}
mpl.rcParams.update(nice_fonts)

sns.set_style()
np.set_printoptions(formatter={'float': "{:0.3f}".format})
### End ###

def set_size(width, fraction=1, subplots=(1, 1)):
    # https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    Sets figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'rpe':
        width_pt = 397.48499
    elif width == 'neurips22':
        width_pt = 397.48499
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)