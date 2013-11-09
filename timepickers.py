"""Methods for choosing times to fold on

This module defines ways of choosing trials or choosing times to fold
spikes or events on.

This is where the pipeline definition is for now as well

I'll try to keep this module agnostic to the parameters of any specific task,
but note that it uses the variables `events` and `trials_info` as loaded by 
the io module.
"""

import numpy as np
from base import Folded
import utility

class TrialPicker:   
    """Object for picking trial numbers based on trials_info"""
    @classmethod
    def pick(self, trials_info, labels=None, label_kwargs=None, 
        **all_kwargs):
        """Returns list of trial numbers satisfying list of constraints.
        
        trials_info : DataFrame containing information about each trial,
            indexed by trial_number
        labels : list of length N, labeling each set of constraints
        label_kwargs : list of length N, consisting of kwargs to pass to
            panda_pick on trials_info (the constraints)
        all_kwargs : added to each label_kwarg
        
        Returns: list of length N
        Each entry is a tuple (label, val) where label is the constraint
        label and val is the picked trial numbers satisfying that constraint.
        
        Example
        labels = ['LB', 'PB']
        label_kwargs = [{'block':2}, {'block':4}]
        all_kwargs = {'outcome':'hit'}
        The return value would be:
        [('LB', list_of_LB_trials), ('PB', list_of_PB_trials)]
        """
        if labels is None:
            labels = self.labels
        if label_kwargs is None:
            label_kwargs = self.label_kwargs
        
        assert len(labels) == len(label_kwargs)
        
        res = []
        for label, kwargs in zip(labels, label_kwargs):
            kk = kwargs.copy()
            kk.update(all_kwargs)
            val = utility.panda_pick(trials_info, **kk)
            res.append((label, val))
        
        return res
    


class IntervalTimePickerNoTrial:
    """Given event names and events structure, return times to lock on"""
    @classmethod
    def pick_d(self, names, events):
        # Define start and stop times of each event
        folded, starts, stops = {}, {}, {}
        for statename in names:
            starts[statename], stops[statename] = \
                IntervalTimePickerNoTrial.pick_one(events, statename)
        
        return starts, stops
    
    @classmethod
    def pick_one(self, events, statename):
        return find_events(events, statename + '_in', statename+'_out')


class EventTimePicker:
    """Given event name and folded events_info, returns times to lock on"""
    @classmethod
    def pick(self, event_name, trials_l):
        """Return df[df.event==event_name] for df in trials_l
        
        If there is no such event, a warning is printed and the trial
        is skipped. If there is more than one event, a warning is taken
        and the first such event is taken.
        """
        res = []
        w, w2 = False, False
        for trial in trials_l:
            val = utility.panda_pick_data(trial, event=event_name).time
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


class TrialsInfoTimePicker:
    """Picks times from columns in trials_info, rather than by event name
    
    Also uses a different, more general syntax. Currently works only with
    pipeline and not pipeline_overblock_oneevent
    """
    def __init__(self, trials_info=None):
        self.trials_info = trials_info
    
    def pick(self, trial_numbers, event_name='stim_onset'):
        """Returns times of event on specified trials
        
        event_name must be a column in trials_info
        Trials that do not exist will be silently dropped!
        """
        # Drop those that don't exist
        idxs = np.asarray(trial_numbers)
        mask = np.in1d(idxs, np.asarray(self.trials_info.index))
        
        # Index and return
        return self.trials_info[event_name][idxs].values



# Other time-picking methods
# Actually not using these at the moment but they may come in handy later
def find_events(events, start_name, stop_name=None, t_start=None, t_stop=None):
    """Given event structure, return epochs around events of specified name.
    
    events : events data frame
    start_name : name of event of that indicates start of epoch
    stop_name : name of event that indicates end of epoch
        if None, then specify t_stop
    t_start : added to time of starting event
    t_stop : added to time of ending event, or if None, then added to time
        of starting event
    """
    starts = np.asarray(events[events.event == start_name].time)
    stops = None
    if stop_name is not None:
        stops = np.asarray(events[events.event == stop_name].time)

        # Correct for some edge effects
        if len(stops) > len(starts):
            # got an extra stop without a start at the beginning
            assert np.all(stops[0] < starts)
            stops = stops[1:]
        elif len(starts) > len(stops):
            # got an extra start without a stop at the end
            assert np.all(starts[-1] > stops)
            starts = starts[:-1]
        
        # Error check
        if np.any(stops < starts):
            # this shouldn't happen after correcting previous error case
            1/0
        
        # Account for some deltas
        if t_start is not None:
            starts = starts + t_start
        if t_stop is not None:
            stops = stops + t_stop
    else:
        assert t_stop is not None
        stops = starts + t_stop
    return starts, stops

def split_events_by_state_name(events, split_state, subtract_off_center=False,
    **kwargs):
    """Divides up events based on a state name that you specify
    
    Returns Folded, split on split_state
    """
    starts = np.asarray(events[events.event == split_state].time)
    res = Folded.from_flat(flat=events, starts=starts, 
        subtract_off_center=subtract_off_center, **kwargs)
    
    return res