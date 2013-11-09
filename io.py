"""Methods for loading non-spike data"""
import os.path
import pandas
import numpy as np, warnings

def load_events(basename, min_time=None, max_time=None):
    """Load events and times from `events` file.
    
    Note: this function defines the format spec for this file.
    
    Returns events as a pandas DataFrame with two columns: `event` and `time`.
    
    If min_time and/or max_time are not None, events outside of this time
    will be discarded.
    
    These will be in whatever timebase the 'events' file uses, currently
    neural time.
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

def load_trials_info(basename, join_on_trial_timings=True, drop_munged=True,
    join_on_trial_numbers=False):
    """Load the trials info matrix. 
    
    Loads a matrix of info about each trial (outcome, stimulus, etc) from
    the specified directory. The index is the behavioral trial number.
    
    join_on_trial_timings : if True, then also load the information from
        the same directory about the timing of certain events in each trial.
        This will drop any trials that didn't actually occur (because they
        were before or after the neural data collection) 
    drop_munged : drop trials that are marked as munged in trial_timings
        No effect if join_on_trials_info is False
    join_on_trial_numbers : this is really just for compatibility reasons
        No effect if join_on_trial_timings is True and trial_timings exists
        But if trial_timings does not exist, this will still allow you to
        filter trials_info for just the ones in trial_numbers and for which
        trials_info.outcome != 'future_trial'
    
    For drop_munged=False, the index of the result is equal to the index 
    of trial_timings.
    trial_timings.index is equal to trial_numbers minus any trials marked
    as 'future_trial' in trials_info. There are only a few corrupted sessions
    where the last entry in trial_numbers is marked as a future trial.
    
    For drop_munged=True, the above still holds except that munged trials
    in trial_numbers and trial_timings will be dropped from the result.
    
    This loading function defines the specification for the trials_info
    and trial_timings csv files.
    """
    trials_info = pandas.read_table(
        os.path.join(basename, 'trials_info'),
        sep=',', index_col=0)
    
    # See about loading trials timing
    tt_fname = os.path.join(basename, 'trial_timings')
    if join_on_trial_timings and not os.path.exists(tt_fname):
        warnings.warn("can't find requested trial_timings in %s" % basename)
        join_on_trial_timings = False
    
    # Optionally do the join
    if join_on_trial_timings:
        # Sometimes weird errors with from_csv?
        trial_timings = pandas.read_table(tt_fname, sep=',', index_col=0)
        
        # Check all trial timings are in trials info (not necessarily v.v.)
        assert np.in1d(list(trial_timings.index), list(trials_info.index)).all()
        
        # Join and optionally demung
        trials_info = trial_timings.join(trials_info)
        
        if drop_munged:
            trials_info = trials_info[~trials_info.is_munged]
            
            if pandas.isnull(trials_info).any().any():
                warnings.warn('demunged trials info contains NaN')
    elif join_on_trial_numbers:
        # Compatibility
        # Keep only "occurred trials"
        # This is mainly for the case where no trial timings are available
        trial_numbers = np.loadtxt(os.path.join(basename, 'TRIAL_NUMBERS'),
            dtype=np.int)
        
        # Filter by the trials that occurred, that is, those for which
        # a behavioral outcome was recorded AND were included in neural
        # database
        trials_info = trials_info[
            (trials_info.outcome != 'future_trial') &
            (trials_info.index.isin(trial_numbers))]
    
    
    return trials_info


