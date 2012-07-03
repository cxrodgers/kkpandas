"""Definition of analysis pipeline.

Select spikes by (session, unit)
Select trials from trials_info with TrialPicker
Select times from trials with EventPicker, events, sync object
Fold over times
Return

However we might wish to iterate over units and/or trial types and/or
times from each trial. That iteration should be handled here, somehow ...

"""

import numpy as np
import io
from timepickers import TrialPicker, EventTimePicker
from ns5_process.RS_Sync import RS_Syncer # remove this dependency
from base import Folded



# something like this should be the analysis pipeline
def pipeline_overblock_oneevent(kkserver, session, unit, rs,
    trial_picker=TrialPicker, trial_picker_kwargs=None,
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