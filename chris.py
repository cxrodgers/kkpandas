"""Wrapper and convenience methods tailored to my experiments"""

import pandas
import io
import numpy as np
import utility
from ns5_process import LBPB
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



def plot_all_stimuli_by_block(binned, consistent_ylim=True):
    """Figure with one subplot for each stimulus and one trace per block"""
    f, axa = plt.subplots(2, 2, figsize=(10,10))

    ymaxes = []

    # Iterate over stimuli pairs
    for spy, lb_stim in enumerate(['le', 'ri']):
        for spx, pb_stim in enumerate(['hi', 'lo']):
            # Get the axis for this stimulus
            ax = axa[spx, spy]
            
            # Get the column names in binned
            lname = '%s_%s_lc' % (lb_stim, pb_stim)
            pname = '%s_%s_pc' % (lb_stim, pb_stim)
            
            ax.plot(binned.t, binned.rate[lname], label='LB', color='b')
            ax.plot(binned.t, binned.rate[pname], label='PB', color='r')
            ax.set_title('%s (blue) vs %s (red)' % (lname, pname))
            
            ymaxes.append(ax.get_ylim()[1])

    for ax in axa.flatten():
        ax.set_ylim((0, np.max(ymaxes)))

    return f
