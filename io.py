"""Methods for loading non-spike data"""
import os.path
import pandas

def load_events(basename):
    """Defines events format spec"""
    events = pandas.read_table(
        os.path.join(basename, 'events'), index_col=None,
        names=['event', 'time'])
    
    return events

def load_trials_info(basename):
    """Defines trials_info format spec"""
    trials_info = pandas.read_table(
        os.path.join(basename, 'trials_info'),
        sep=',', index_col=0)
    
    return trials_info

