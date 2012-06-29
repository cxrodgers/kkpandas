import matplotlib.pyplot as plt


def plot_binned_by_level(binned, ax=None):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    topkeys = list(binned.columns.levels[0])
    lsl = ['-', ':', '.-']
    for n, topkey in enumerate(topkeys):
        plot_binned(binned[topkey], ax=ax, ls=lsl[n])
    
    ax.legend(binned.columns, loc='best')
    plt.show()
    return ax

def plot_binned(binned, ax=None, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    csl = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray']
    for n, (label, col) in enumerate(binned.rate.iteritems()):
        if 'color' in kwargs:
            ax.plot(binned.t, col.values, label=label, **kwargs)
        else:
            ax.plot(binned.t, col.values, color=csl[n], label=label, **kwargs)
    
    ax.legend(loc='best')
    plt.show()
    return ax

def plot_psth_with_rasters_from_dict(
    dfolded, keys=None, spshape=None, bins=None):
    
    if keys is None:
        keys = dfolded.keys()
    
    f = plt.figure()
    for n, key in enumerate(keys):
        folded = dfolded[key]
        ax = f.add_subplot(spshape[0], spshape[1], n + 1)        
        plot_psth_with_rasters(folded, bins=bins, ax=ax)
    
    plt.show()

def plot_psth_with_rasters(folded, smoothed=None, bins=None, ax=None):
    myutils.plot_rasters(folded, ax=ax)
    
    if smoothed is None:
        smoothed = Binned.from_folded(folded, bins=bins)
    ax.plot(smoothed.t, smoothed.rate)
    
