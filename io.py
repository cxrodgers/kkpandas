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




# Ugly IO functions to be moved elsewhere
def get_events_and_trials(dirname, bskip=0):
    """Returns times of events and trials info.
    
    Put this in Recording Session!
    
    Also, auto-detect bskip from TRIAL_NUMBER
    
    Given a directory containing the following files:
        * some bcontrol data
        * TIMESTAMPS (neural onsets)
    
    bskip : Number of behavioral trials that occurred before neural
        recording began.
    
    Will load all of the events and trials, convert to a neural timebase.
    
    This should go into RecordingSession, then dump the formatted
    DataFrames there.
    
    Returns:
        events, trials_info
    """
    # get events
    bcld = bcontrol.Bcontrol_Loader_By_Dir(dirname)
    bcld.load()
    peh = bcld.data['peh']
    TI = bcontrol.demung_trials_info(bcld)
    events = pandas.DataFrame.from_records(
        bcontrol.generate_event_list(peh, bcld.data['TRIALS_INFO'], bskip))

    # store sync info
    # this should be done by RecordingSession and then stored for future use

    btimes = events[events.event == 'play_stimulus_in'].time[bskip:]
    ntimes = np.loadtxt(os.path.join(dirname, 'TIMESTAMPS')) / 30e3
    b2n = np.polyfit(btimes, ntimes, deg=1)

    # Convert behavioral times
    events['time'] = np.polyval(b2n, events.time)

    # Insert trial times into trials_info
    TI.set_index('trial_number', inplace=True)
    TI.insert(0, 'time', np.nan)
    TI['time'][bskip:bskip+len(ntimes)] = ntimes

    return events, TI
