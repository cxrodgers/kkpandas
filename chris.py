"""Wrapper and convenience methods tailored to my experiments"""

import pandas
import io
import numpy as np
import analysis
import LBPB
from ns5_process.RS_Sync import RS_Syncer
from analysis import Folded

# TRIALS_INFO picking functions with my defaults
def pick_trial_numbers(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial numbers satisfying condition
    
    This convenience method provides common defaults for me
    isnotnull : asserts that the provided column is not Null
        By convention, returns those with non-null times
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




# Column name mappers from one format to another
def names2multilevel(df):
    """Inplace reorder/rename"""
    df.reorder(LBPB.mixed_stimnames)
    df.rename(LBPB.stimname2block_sound_tuple)
    
    return df



class BlockTrialPicker:
    """Given trials_info, returns (label, trial_numbers) for blocks"""
    @classmethod
    def pick(self, trials_info):
        keys = ('LB', 'PB')
        kwargs_l = ({'block': 2}, {'block': 4})
        meth = pick_trial_numbers
        
        res = []
        for key, kwargs in zip(keys, kwargs_l):
            res.append((key, meth(trials_info, isnotnull=None, **kwargs)))
        return res

class EventTimePicker:
    """Given event name and folded events_info, returns times to lock on"""
    @classmethod
    def pick(self, event_name, trials_l):
        res = []
        for trial in trials_l:
            val = analysis.panda_pick_data(trial, event=event_name).time.item()
            res.append(val)
        return res

# something like this should be the analysis pipeline
def pipeline_overblock_oneevent(kkserver, session, unit, rs,
    trial_picker=BlockTrialPicker,
    evname='play_stimulus_in'):
    
    # And a trial_server object?
    trials_info = io.load_trials_info(rs.full_path)
    events = io.load_events(rs.full_path)

    # Spike selection
    spikes = np.asarray(
        kkserver.load(session=session, unit=unit).spike_time)

    
    # Select trials from behavior
    picked_trials_l = BlockTrialPicker.pick(trials_info)
    
    # Fold events structure on start times
    rss = RS_Syncer(rs)
    f = Folded.from_flat(flat=events, starts=rss.trialstart_nbase, 
        subtract_off_center=False)
    
    # Convert to dict Folded representation with trial numbers as labels
    tn2ev = dict(zip(rss.btrial_numbers, f))
    
    # Take advantage of the fact that trial numbers are the link between
    # picked_trials_l and tn2ev to
    # Iterate over picked_trials_l and extract time from each trial
    label2timelocks = {}
    for label, trial_numbers in picked_trials_l:
        # Extract trials by number (put this as accessor method in Folded
        trials = [tn2ev[tn] for tn in trial_numbers] # go in folded
    
        # Get timelock times by applying a function to each entry (put this in Folded)
        times = EventTimePicker.pick(evname, trials)
        
        label2timelocks[labe] = times
    
    # Now fold out over timelocked times
    res = {}
    for label, timelocks in label2timelocks.items():
        # Now fold spike times on timelock times    
        res[label] = Folded.from_flat(
            flat=spikes, centers=timelocks, dstart=-.25,
            dstop=.25)

    return res