"""Module for operating on RecordingSession objects using kkpandas tools"""
import kkpandas
import kkio
import numpy as np
import matplotlib.pyplot as plt
import os
from base import Folded
import plotting

def RS_plot_tuning_curve(rs, bins, savefig=True, figsize=(15,15), **folding_kwargs):
    """Folds data for an RS and plots the results as separate tuning curves.
    
    """
    ddfolded, tc_freqs, tc_attens, tc_freq_labels, tc_atten_labels =\
        RS_fold_for_tuning_curve(rs, **folding_kwargs)
    
    if savefig is True:
        savefig = rs.full_path

    for group in sorted(ddfolded.keys()):
        dfolded = ddfolded[group]
        f = plot_tuning_curve(dfolded, tc_freq_labels, tc_atten_labels, 
            bins=bins, figsize=figsize)
        f.suptitle('Group %d' % group)
        
        if savefig:
            f.savefig(os.path.join(savefig, 'tuning_curve_%s_group_%d.png' % (
                rs.session_name, group)))
            plt.close(f)
    

def RS_fold_for_tuning_curve(rs, override_dir=None, **bin_kwargs):
    """Fold RS for tuning curve.
    
    This wrapper extracts info from rs, then calls fold_for_tuning_curve
    on each group. It checks for consistency of the binning across groups.
    
    Returns:
        ddfolded, all_tc_freqs, all_tc_attens
    
        ddfolded: dict {group: dfolded}
        all_tc_freqs, all_tc_attens: consistent bins across groups
    """
    if override_dir is None:
        override_dir = rs.last_klusters_dir()
    
    # Stuff that is the same for all groups
    timestamps = rs.read_timestamps() / rs.get_sampling_rate()
    tones = np.loadtxt(os.path.join(rs.full_path, 'tones'))
    attens = np.loadtxt(os.path.join(rs.full_path, 'attens'), dtype=np.int)    
    assert len(tones) == len(attens)
    
    # Deal with alignment issues    
    if len(timestamps) > len(tones):
        print "warning: too many timestamps, truncating"
        timestamps = timestamps[:len(tones)]
    if len(tones) > len(timestamps):
        print "warning: not enough timetsamps, truncating tones"
        tones = tones[:len(timestamps)]
        attens = attens[:len(timestamps)]
    
    # Return variables
    all_tc_freqs, all_tc_attens, all_tc_freq_labels, all_tc_atten_labels = \
        None, None, None, None
    res = {}    

    # Iterate through groups
    kfs = kkpandas.KKFileSchema(override_dir)
    for group in kfs.groups:
        # Get spikes from this group
        spikes = kkio.from_KK(override_dir, groups_to_get=[group], 
            fs=rs.get_sampling_rate())
        spikes = np.sort(np.asarray(spikes.time))

        # Fold on stimulus types
        dfolded, tc_freqs, tc_attens, tc_freq_labels, tc_atten_labels = \
            fold_for_tuning_curve(spikes, timestamps, tones, attens,
                **bin_kwargs)
        
        # Store results
        if all_tc_freqs is None:
            all_tc_freqs = tc_freqs
        else:
            assert np.all(tc_freqs == all_tc_freqs)
        
        if all_tc_attens is None:
            all_tc_attens = tc_attens
        else:
            # Because inf
            assert np.all(np.abs(tc_attens[:-1] - all_tc_attens[:-1]) < 1e-6)
        
        if all_tc_freq_labels is None:
            all_tc_freq_labels = tc_freq_labels
        else:
            assert np.all(tc_freq_labels == all_tc_freq_labels)
        
        if all_tc_atten_labels is None:
            all_tc_atten_labels = tc_atten_labels
        else:
            assert np.all(np.abs(tc_atten_labels - all_tc_atten_labels) < 1e-6)
        
        res[group] = dfolded

    return res, all_tc_freqs, all_tc_attens, all_tc_freq_labels, all_tc_atten_labels
    
    

# Utility functions that don't know about RS
def fold_for_tuning_curve(spikes, timestamps, tones, attens,
    tc_freqs=None, tc_attens=None, freq_min=None, freq_max=None,
    n_freq_bins=None, dstart=-.05, dstop=.14):
    """Fold spikes into freq/atten bins for tuning curve
    
    spikes : times in seconds, I will sort
    timestamps : time in seconds
    tones : frequency of stimulus, same shape as timestamps
    attens : attenuation of stimulus, same shape as timestamps
    
    tc_freqs, tc_attens : bin edges
        If None, will generate from freq_min, ferq_max, n_freq_bins
    
    dstart, dstop: seconds of time around each timestamp
    
    Returns:
        dfolded, tc_freqs, tc_attens, tc_freq_labels, tc_atten_labels
        dfolded : dict of Folded, keyed by index (fb, ab) into freq and 
            atten labels
    """
    # Set up bin edges ... one more than the number of bins
    if tc_freqs is None:    
        tc_freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), 
            n_freq_bins + 1)
    if tc_attens is None:
        tc_attens = np.concatenate([np.sort(np.unique(attens)), [np.inf]])

    # Labels of the bins, ie bin "centers"
    tc_freq_labels = 10 ** (
        np.log10(tc_freqs[:-1]) + np.diff(np.log10(tc_freqs)) / 2)
    tc_atten_labels = tc_attens[:-1]

    # Place each stimulus in a bin
    # These will range from (0, len(edges) - 1)
    tone_freq_bin = np.searchsorted(tc_freqs, tones) - 1
    tone_atten_bin = np.searchsorted(tc_attens, attens) 

    # Sort spikes for folding
    spikes = np.sort(spikes)

    # Create folded for each bin
    dfolded = {}
    for fb in range(len(tc_freq_labels)):
        for ab in range(len(tc_atten_labels)):
            seln = ((tone_freq_bin == fb) & (tone_atten_bin == ab))        
            dfolded[(fb, ab)] = Folded.from_flat(spikes, 
                centers=timestamps[seln], dstart=dstart, dstop=dstop)
    
    return dfolded, tc_freqs, tc_attens, tc_freq_labels, tc_atten_labels

def plot_tuning_curve(dfolded, tc_freq_labels, tc_atten_labels, bins, figsize=(15,15)):
    """Plots the output from fold_for_tuning_curve"""
    # Plot each bin, with frequency along x (increasing to right) and attenuation
    # along y (increasing to bottom, that is, volume decreasing to bottom)
    f, axa = plt.subplots(len(tc_atten_labels), len(tc_freq_labels),
        figsize=figsize, squeeze=False)
    for fb in range(len(tc_freq_labels)):
        for ab in range(len(tc_atten_labels)):
            ax = axa[ab, fb]
            plotting.plot_psth_with_rasters(
                dfolded[(fb, ab)], ax=ax, bins=bins)

            # Label the left column
            if fb == 0:
                ax.set_title('-%ddB' % tc_atten_labels[ab])
            
            # Label the top row
            if ab == 0:
                ax.set_title('%0.1fK' % (tc_freq_labels[fb] / 1000.))
    return f