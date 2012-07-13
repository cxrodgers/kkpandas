"""Wrapper and convenience methods tailored to my experiments"""

import pandas
import io
import numpy as np
import utility
from ns5_process import LBPB
from ns5_process.RS_Sync import RS_Syncer
from base import Folded
from timepickers import TrialPicker
import matplotlib.pyplot as plt
import plotting

# TRIALS_INFO picking functions with my defaults
def pick_trial_numbers(trials_info, outcome='hit', nonrandom=0, 
    isnotnull=None, **kwargs):
    """Returns trial numbers satisfying condition
    
    This convenience method provides common defaults for me
    isnotnull : asserts that the provided column is not Null
    """
    return utility.panda_pick(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs)

def pick_trials(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial rows satisfying condition
    
    This convenience method provides common defaults for me
    """
    return utility.panda_pick_data(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs)

def pick_trial_times(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial times satisfying condition
    
    This convenience method provides common defaults for me
    """
    return np.asarray(utility.panda_pick_data(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs).time)




# Column name mappers from one format to another
def names2multilevel(df):
    """Inplace reorder/rename"""
    df.reorder(LBPB.mixed_stimnames)
    df.rename(LBPB.stimname2block_sound_tuple)
    
    return df



class BlockTrialPicker(TrialPicker):
    """Given trials_info, returns (label, trial_numbers) for blocks"""
    labels = ('LB', 'PB')
    label_kwargs = ({'block': 2}, {'block': 4})

    @classmethod
    def _pick(self, *args, **kwargs):
        return pick_trial_numbers(*args, **kwargs)

class StimulusTrialPicker(TrialPicker):
    """Given trials_info, returns (label, trial_numbers) for blocks"""
    labels = LBPB.stimnames
    label_kwargs = tuple([{'stim_name' : sn} for sn in LBPB.stimnames])

    @classmethod
    def _pick(self, *args, **kwargs):
        return pick_trial_numbers(*args, **kwargs)



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

def plot_tuning_curve(dfolded, tc_freq_labels, tc_atten_labels, bins):
    """Plots the output from fold_for_tuning_curve"""
    # Plot each bin, with frequency along x (increasing to right) and attenuation
    # along y (increasing to bottom, that is, volume decreasing to bottom)
    f, axa = plt.subplots(len(tc_atten_labels), len(tc_freq_labels))
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