"""Methods for choosing times to fold on"""

import numpy as np
from analysis import Folded

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