import matplotlib.pyplot as plt
import numpy as np
from base import Binned

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

def plot_binned(binned, units=None, ax=None, legend=True, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(111)
    
    csl = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gray']
    
    for n, (label, col) in enumerate(binned.rate_in(units).iteritems()):
        if 'color' in kwargs:
            ax.plot(binned.t, col.values, label=label, **kwargs)
        else:
            color = csl[np.mod(n, len(csl))]
            ax.plot(binned.t, col.values, color=color, label=label, **kwargs)
    
    if legend:
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
    ax = plot_rasters(folded, ax=ax)
    
    if smoothed is None:
        smoothed = Binned.from_folded(folded, bins=bins)
    ax.plot(smoothed.t, smoothed.rate)
    

def plot_rasters(folded_spike_times, ax=None, full_range=1.0, 
    y_offset=0.0, plot_kwargs=None, sort_by_duration=False):
    """Plots raster of spike times or psth object.
    
    folded_spike_times : Folded, or any list of arrays of time-locked spike times
    ax : axis object to plot into
    plot_kwargs : any additional plot specs. Defaults:
        if 'color' not in plot_kwargs: plot_kwargs['color'] = 'k'
        if 'ms' not in plot_kwargs: plot_kwargs['ms'] = 4
        if 'marker' not in plot_kwargs: plot_kwargs['marker'] = '|'
        if 'ls' not in plot_kwargs: plot_kwargs['ls'] = 'None'    
    full_range: y-value of top row (last trial), default 1.0
    
    Returns the axis in which it was plotted
    """
    if sort_by_duration:
        starts, stops = folded_spike_times.starts, folded_spike_times.stops
        durations = stops - starts
        idxs = np.argsort(durations)
        
        folded_spike_times = np.asarray(folded_spike_times.values)[idxs]
        starts, stops = starts[idxs], stops[idxs]
    
    # build axis
    if ax is None:
        f = plt.figure(); ax = f.add_subplot(111)
    
    # plotting defaults
    if plot_kwargs is None:
        plot_kwargs = {}
    if 'color' not in plot_kwargs: plot_kwargs['color'] = 'k'
    if 'ms' not in plot_kwargs: plot_kwargs['ms'] = 4
    if 'marker' not in plot_kwargs: plot_kwargs['marker'] = '|'
    if 'ls' not in plot_kwargs: plot_kwargs['ls'] = 'None'
    
    if full_range is None:
        full_range = float(len(folded_spike_times))
    
    for n, trial_spikes in enumerate(folded_spike_times):
        ax.plot(trial_spikes, 
            y_offset + np.ones(trial_spikes.shape, dtype=np.float) * 
            n / float(len(folded_spike_times)) * full_range,
            **plot_kwargs)
        
        if sort_by_duration:
            ax.plot([starts[n]-starts[n]], 
                y_offset + n / float(len(folded_spike_times)) * full_range, 'ro')
            ax.plot([stops[n]-starts[n]], 
                y_offset + n / float(len(folded_spike_times)) * full_range, 'bo')                
    
    return ax