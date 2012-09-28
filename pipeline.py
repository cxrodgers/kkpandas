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
    evname='play_stimulus_in', folding_kwargs=None, sort_spikes=True,
    final_folded_map=None, final_folded_map_dtype=np.int,
    label_with_btrial_numbers=True):
    """This aims to be the all-encompassing pipeline
    
    See IntervalPipeline for a different design philosophy.

    Each 'category' of trials is folded together.
    
    trial_picker_kwargs : dict
        Definition of each category. It has the following items:
        'labels' : list
            Name of each category (keys in returned dict)
        'label_kwargs' : list, of same length as 'labels'
            Definition of each category. Passed as keyword arguments to
            `trial_picker`. For the default picker: each key, value pair is applied
            to trials_info to select trials for this category. For example,
            {'outcome': 'hit', 'block': 2}
            
            This can also be a MultiIndex with suitably defined attribute
            `names`. That way actually makes more sense to me in the long 
            run. For now it is converted to the above dict-like syntax.
        Any other key, value pairs in this dict are passed to `trial_picker`
        for EVERY category. Ex: {'nonrandom': 0}

    trial_picker : object
        Object that picks the trials for each category, according to
        `trial_picker_kwargs` and using TRIALS_INFO
    
    label_with_btrial_numbers : bool
        If True, then an attribute called 'labels' is stored in each returned
        Folded. 'labels' is the behavioral trial number of each entry in
        the Folded.
    
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
    
    # Have to sort them if they aren't already
    if sort_spikes:
        spikes = np.sort(spikes)
    
    # Convert trial_picker_kwargs from MultiIndex if necessary
    if hasattr(trial_picker_kwargs['label_kwargs'], 'names'):
        trial_picker_kwargs = trial_picker_kwargs.copy()
        mi = trial_picker_kwargs['label_kwargs']
        trial_picker_kwargs['label_kwargs'] = [
            dict([(name, val) for name, val in zip(mi.names, val2)]) 
            for val2 in list(mi)]
    
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
    label2timelocks, label2btrial_numbers = {}, {}    
    for label, trial_numbers in picked_trials_l:
        # Extract trials by number (put this as accessor method in Folded
        trials = [tn2ev[tn] for tn in trial_numbers] # go in folded
    
        # Store the trial numbers that we picked (for labeling the Folded later)
        label2btrial_numbers[label] = np.asarray(trial_numbers)
    
        # Get timelock times by applying a function to each entry (put this in Folded)
        times = EventTimePicker.pick(evname, trials)
        
        # Store the timelock times
        label2timelocks[label] = times
    
    # Now fold out over timelocked times
    if folding_kwargs is None: 
        folding_kwargs = {}
    res = {}    
    for label, timelocks in label2timelocks.items():
        # Now fold spike times on timelock times    
        res[label] = Folded.from_flat(
            flat=spikes, centers=timelocks, **folding_kwargs)
        
        # Optionally label each trial
        # This is sort of a hack ... eventually would like to be able
        # to access the trials by trial number instead of index
        if label_with_btrial_numbers:
            if hasattr(f, 'labels'):
                print "warning: overwriting something in Folded"
            res[label].labels = label2btrial_numbers[label]
        
        # Optionally apply a map to each folded
        if final_folded_map is not None:
            res[label] = np.asarray(map(final_folded_map, res[label]),
                dtype=final_folded_map_dtype)

    return res