"""Wrapper and convenience methods tailored to my experiments"""

import pandas
import numpy as np
import analysis
import LBPB

# TRIALS_INFO picking functions with my defaults
def pick_trial_numbers(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial numbers satisfying condition
    
    This convenience method provides common defaults for me
    isnotnull : asserts that the provided column is not Null
        By convention, returns those with non-null times
    """
    return analysis.panda_pick(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs)

def pick_trials(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial rows satisfying condition
    
    This convenience method provides common defaults for me
    """
    return analysis.panda_pick_data(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs)

def pick_trial_times(trials_info, outcome='hit', nonrandom=0, 
    isnotnull='time', **kwargs):
    """Returns trial times satisfying condition
    
    This convenience method provides common defaults for me
    """
    return np.asarray(analysis.panda_pick_data(trials_info, outcome=outcome, 
        nonrandom=nonrandom, isnotnull=isnotnull, **kwargs).time)




# Column name mappers from one format to another
def names2multilevel(df):
    """Inplace reorder/rename"""
    df.reorder(LBPB.mixed_stimnames)
    df.rename(LBPB.stimname2block_sound_tuple)
    
    return df



# something like this should be the analysis pipeline
def pipeline(unit):
    # Find data
    subdir = os.path.join(data_dir, unit.xpath('../../../@session_name')[0])
    rs = RecordingSession.RecordingSession(subdir)
    basename = rs.last_klusters_dir()
    
    # This could be put into a KK_server object
    # You pass session name, group, and unit (or fully specified unit)
    # It returns spikes and deals with memoization, flushing, etc
    unum = int(unit.find('group').text) * 100 + int(unit.find('cluster').text)
    spikes = kkpandas.from_KK(basename=basename, load_memoized=True,
        save_memoized=True, group_multiplier=100, fs=30e3)
    
    # Spike selection
    sspikes = np.asarray(spikes[spikes.unit == unum].spike_time)
    
    # And a trial_server object?
    trials_info = kkpandas.io.load_trials_info(rs.full_path)
    events = kkpandas.io.load_events(rs.full_path)
    
    # Select trials from behavior
    LB_trial_numbers = kkpandas.chris.pick_trial_numbers(trials_info,
        isnotnull=None, block=2)
    PB_trial_numbers = kkpandas.chris.pick_trial_numbers(trials_info,
        isnotnull=None, block=4)
    
    # Fold events structure on start times
    rss = RS_Syncer(rs)
    f = kkpandas.Folded.from_flat(flat=events, starts=rss.trialstart_nbase, 
        subtract_off_center=False)
    
    # Convert to dict Folded representation with trial numbers as labels
    tn2ev = dict(zip(rss.btrial_numbers, f))
    
    # Extract trials by number (put this as accessor method in Folded
    LB_trials = [tn2ev[tn] for tn in LB_trial_numbers] # go in folded
    PB_trials = [tn2ev[tn] for tn in PB_trial_numbers] # go in folded
    
    # Get timelock times by applying a function to each entry (put this in Folded)
    LB_times = map(lambda ev: kkpandas.analysis.panda_pick_data(
        ev, event='play_stimulus_in').time.item(),
        LB_trials)
    PB_times = map(lambda ev: kkpandas.analysis.panda_pick_data(
        ev, event='play_stimulus_in').time.item(),
        PB_trials)

    # Now fold spike times on timelock times
    d = {}
    for label, times in zip(['LB', 'PB'], [LB_times, PB_times]):
        d[label] = kkpandas.Folded.from_flat(
            flat=sspikes, centers=times, dstart=-.25,
            dstop=.25)
    ures[(rs.session_name, unum)] = d    