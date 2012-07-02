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
    isnotnull=None, **kwargs):
    """Returns trial numbers satisfying condition
    
    This convenience method provides common defaults for me
    isnotnull : asserts that the provided column is not Null
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


class TrialPicker:    
    @classmethod
    def pick(self, trials_info, labels=None, label_kwargs=None, 
        **all_kwargs):
        if labels is None:
            labels = self.labels
        if label_kwargs is None:
            label_kwargs = self.label_kwargs
        
        assert len(labels) == len(label_kwargs)
        
        res = []
        for label, kwargs in zip(labels, label_kwargs):
            kk = kwargs.copy()
            kk.update(all_kwargs)
            val = self._pick(trials_info, **kk)
            res.append((label, val))
        
        return res
    
    @classmethod
    def _pick(self, *args, **kwargs):
        return pick_trial_numbers(*args, **kwargs)

class BlockTrialPicker(TrialPicker):
    """Given trials_info, returns (label, trial_numbers) for blocks"""
    labels = ('LB', 'PB')
    label_kwargs = ({'block': 2}, {'block': 4})

class StimulusTrialPicker(TrialPicker):
    """Given trials_info, returns (label, trial_numbers) for blocks"""
    labels = LBPB.stimnames
    label_kwargs = tuple([{'stim_name' : sn} for sn in LBPB.stimnames])


class EventTimePicker:
    """Given event name and folded events_info, returns times to lock on"""
    @classmethod
    def pick(self, event_name, trials_l):
        res = []
        w, w2 = False, False
        for trial in trials_l:
            val = analysis.panda_pick_data(trial, event=event_name).time
            if len(val) > 1:
                w2 = True
                res.append(val.values[0])
            elif len(val) == 0:
                w = True
            else:
                res.append(val.item())
        
        if w:
            print "warning: some events did not occur"
        if w2:
            print "warning: multiple events detected on some trials"
        return res

# something like this should be the analysis pipeline
def pipeline_overblock_oneevent(kkserver, session, unit, rs,
    trial_picker=BlockTrialPicker, trial_picker_kwargs=None,
    evname='play_stimulus_in', folding_kwargs=None):
    
    # And a trial_server object?
    trials_info = io.load_trials_info(rs.full_path)
    events = io.load_events(rs.full_path)

    # Spike selection
    spikes = np.asarray(
        kkserver.load(session=session, unit=unit).spike_time)

    
    # Select trials from behavior
    if trial_picker_kwargs is None:
        trial_picker_kwargs = {}
    picked_trials_l = trial_picker.pick(trials_info, **trial_picker_kwargs)
    
    # Fold events structure on start times
    rss = RS_Syncer(rs)
    f = Folded.from_flat(flat=events, starts=rss.trialstart_nbase, 
        subtract_off_center=False)
    
    # Convert to dict Folded representation with trial numbers as labels
    tn2ev = dict(zip(rss.btrial_numbers, f))
    
    # Here is the link between behavior and neural
    # We have picked_trials_l, a list of trial numbers selected from behavior
    # And tn2ev, a dict keyed on trial numbers that actually occurred in
    # neural recording
    # We need to pick out events from each of the trials in each category
    # But first we need to drop trials that never actually occurred from
    # pick_trials_l
    for n in range(len(picked_trials_l)):
        picked_trials_l[n] = (
            picked_trials_l[n][0],
            picked_trials_l[n][1][
            np.in1d(picked_trials_l[n][1], rss.btrial_numbers)])

    # Iterate over picked_trials_l and extract time from each trial
    label2timelocks = {}
    for label, trial_numbers in picked_trials_l:
        # Extract trials by number (put this as accessor method in Folded
        trials = [tn2ev[tn] for tn in trial_numbers] # go in folded
    
        # Get timelock times by applying a function to each entry (put this in Folded)
        times = EventTimePicker.pick(evname, trials)
        
        label2timelocks[label] = times
    
    # Now fold out over timelocked times
    if folding_kwargs is None: 
        folding_kwargs = {}
    res = {}    
    for label, timelocks in label2timelocks.items():
        # Now fold spike times on timelock times    
        res[label] = Folded.from_flat(
            flat=spikes, centers=timelocks, **folding_kwargs)

    return res