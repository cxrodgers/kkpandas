"""Methods for loading non-spike data"""
import os.path
import pandas

def load_events(basename, min_time=None, max_time=None):
    """Load events and times from `events` file.
    
    Note: this function defines the format spec for this file.
    
    Returns events as a pandas DataFrame with two columns: `event` and `time`.
    
    If min_time and/or max_time are not None, events outside of this time
    will be discarded.
    """
    events = pandas.read_table(
        os.path.join(basename, 'events'), index_col=None,
        names=['event', 'time'])
    
    if min_time is not None:
        events = events[events.time >= min_time]
        events.index = range(len(events))
    if max_time is not None:
        events = events[events.time < max_time]
        events.index = range(len(events))
    
    return events

def load_trials_info(basename):
    """Defines trials_info format spec"""
    trials_info = pandas.read_table(
        os.path.join(basename, 'trials_info'),
        sep=',', index_col=0)
    
    return trials_info

