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


