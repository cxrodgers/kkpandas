

# PLOTTING stuff
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
    
