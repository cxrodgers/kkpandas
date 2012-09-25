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
from timepickers import TrialPicker, EventTimePicker, IntervalTimePickerNoTrial
# Something problematic about this circular import
# Doesn't belong here anyway since this should not depend on RS_Syncer
# Will move to into that pipeline method till I figure this out
#from ns5_process.RS_Sync import RS_Syncer # remove this dependency
from base import Folded

class IntervalPipeline:
    def __init__(self, spike_server=None, 
        time_picker=IntervalTimePickerNoTrial):
        
        
        self.spike_server = spike_server
        self.time_picker = time_picker
    
    def run(self, rs=None, interval_names=None, session=None, unit=None):
        if self.spikes is None:
            self.spikes = self.select_spikes(session=session, unit=unit)
        self.events = self.load_events(rs.full_path)
        self.starts_d, self.stops_d = self.select_times(self.events,
            interval_names)
        self.dfolded = self.fold_spikes_on_times(interval_names,
            self.starts_d, self.stops_d)
        
        return self.dfolded

    def select_spikes(self, session=None, unit=None):
        return np.asarray(
            self.spike_server.load(session=session, unit=unit).spike_time)

    def load_events(self, full_path=None):
        return io.load_events(full_path)
    
    def select_times(self, events, interval_names=None, **kwargs):
        # Extract times
        starts_d, stops_d = self.time_picker.pick_d(events=events,
            names=interval_names, **kwargs)
        return starts_d, stops_d
    
    def fold_spikes_on_times(self, interval_names, starts_d, stops_d):
        # For each interval, fold spikes on starts_d and stops_d
        dfolded = {}
        for statename in interval_names:
            dfolded[statename] = Folded.from_flat(
                np.asarray(self.spikes.spike_time),
                starts=starts_d[statename], stops=stops_d[statename])    
    
        return dfolded

# something like this should be the analysis pipeline
def pipeline_overblock_oneevent(kkserver, session, unit, rs,
    trial_picker=TrialPicker, trial_picker_kwargs=None,
    evname='play_stimulus_in', folding_kwargs=None, sort_spikes=True):
    """This aims to be the all-encompassing pipeline
    
    See IntervalPipeline for a different design philosophy.
    
    Example: Bin by block
    # Set up the pipeline
    # How to parse out trials
    trial_picker_kwargs = {
        'labels':['LB', 'PB'], 
        'label_kwargs': [{'block':2}, {'block':4}],
        'outcome': 'hit', 'nonrandom' : 0
        }
    
    # How to fold the window around each trial
    folding_kwargs = {'dstart': -.25, 'dstop': 0.}
    
    # Run the pipeline
    res = kkpandas.pipeline.pipeline_overblock_oneevent(
        kk_server, session, unit2unum(unit), rs, 
        trial_picker_kwargs=trial_picker_kwargs,
        folding_kwargs=folding_kwargs)    
    
    sort_spikes : whether to sort the spike times after loading.
        Certainly the spikes should be sorted before processing
        This defaults to False because it's often the case that they're
        pre-sorted
    """
 
    from ns5_process.RS_Sync import RS_Syncer # remove this dependency
    
    # And a trial_server object?
    trials_info = io.load_trials_info(rs.full_path)
    events = io.load_events(rs.full_path)

    # Spike selection
    spikes = np.asarray(
        kkserver.get(session=session, unit=unit).time)
    
    if sort_spikes:
        spikes = np.sort(spikes)

    
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