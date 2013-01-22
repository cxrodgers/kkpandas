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


# Too ugly ... put this in kkrs
def fold_events_on_trial_starts(rs):
    """Return a Folded of DataFrames of event times, one per trial
    
    The 'state_%d_out' state is detected as the trial boundaries
    It may or may not be included in each entry, depending on floating
    point
    """
    # This object necessary to get neural start times and btrial labels
    # for each trial
    from ns5_process.RS_Sync import RS_Syncer
    import kkpandas

    # Fold events structure on start times
    # This is necessary in order to extract the trial start times
    # for the trials in TRIALNUMBERS only
    rss = RS_Syncer(rs)
    events = load_events(rs.full_path)
    f = kkpandas.Folded.from_flat(flat=events, starts=rss.trialstart_nbase, 
        subtract_off_center=False, labels=rss.btrial_numbers)

    return f