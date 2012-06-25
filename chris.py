"""Wrapper and convenience methods tailored to my experiments"""

import pandas
import numpy as np
import analysis



def pick_trial_numbers(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial numbers satisfying condition
    
    This convenience method provides common defaults for me
    """
    return analysis.panda_pick(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs)

def pick_trials(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial rows satisfying condition
    
    This convenience method provides common defaults for me
    """
    return analysis.panda_pick_data(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs)

def pick_trial_times(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial times satisfying condition
    
    This convenience method provides common defaults for me
    """
    return np.asarray(analysis.panda_pick_data(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs).time)
