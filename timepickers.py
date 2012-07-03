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
    





class EventTimePicker:
    """Given event name and folded events_info, returns times to lock on"""
    @classmethod
    def pick(self, event_name, trials_l):
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
    starts = events[events.event == start_name].time
    stops = None
    if stop_name is not None:
        stops = events[events.event == stop_name].time
        if len(stops) != len(starts):
            # need to drop one from the end
            1/0
        if np.any(stops < starts):
            # this shouldn't happen after correcting previous error case
            1/0
        
        if t_start is not None:
            starts = starts + t_start
        if t_stop is not None:
            stops = stops + t_stop
    else:
        assert t_stop is not None
        stops = starts + t_stop
    return np.asarray(starts), np.asarray(stops)

def split_events_by_state_name(events, split_state, subtract_off_center=False,
    **kwargs):
    """Divides up events based on a state name that you specify
    
    Returns Folded, split on split_state
    """
    starts = np.asarray(events[events.event == split_state].time)
    res = Folded.from_flat(flat=events, starts=starts, 
        subtract_off_center=subtract_off_center, **kwargs)
    
    return res