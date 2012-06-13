""" Convenience methods to analyze spike times in pandas data frame

Need methods to load from disparate data sources into the canonical
structures below, including time base synchronization.

Canonical data structures:
spike data
spike times


event data
event list


trial data
trial_number    105  non-null values (make this indx??)
correct_side    105  non-null values
stim_number     105  non-null values
nonrandom       105  non-null values
outcome         105  non-null values
go_or_nogo      105  non-null values
block           105  non-null values
stim_name       105  non-null values

* Everything should be in the same time base (preferably neural)
   This defines the link between spike data and event data
* Link between trial data and event data is defined by the presence of
    'start_trial_N_out' events, where N equals the entry in 
    trials_info.trial_number (or possibly the index of this DataFrame)
  Perhaps additionally I will insert a new column in trials_info called
  'start_time'


flat = Flat.from_KK(KK_dirname)
events = Flat.from_bcontrol(KK_dirname)
t_starts, t_stops = TrialSlicer(events, start='ambient_in', stop='ambient_out')
folded = Folded.from_flat(flat, t_starts, t_stops)
binned = Binned.from_folded(folded, binwidth=.001)
plot(binned.mean())
"""



import pandas
import numpy as np
from ns5_process import bcontrol
import os.path

class Folded:
    """Stores spike times on each trial timelocked to some event
    
    List-like or dict-like
    
    Provides:
        iteration over trials, each entry being an array of spike times
            (or potentially a spiketrain object?)
        starts : start time of each trial, relative to event
        stops : stop time of each trial, relative to event
        to_binned_by_trial : bin each trial separately
            - actually this should be a method of Binned, since it depends
            on how to smooth
        to_binned : bin all trials together
            - same
    """

class Binned:
    """Stores binned spike counts, optionally across categories (eg, stimuli).
    
    This is useful for comparing categories that have the same time base,
    for example, responses of different neurons to a stimulus. It can accomodate
    time windows of different durations but not different granularities.
    
    Regardless of implementation, should provide:
        edges : edges of each bin
        t   : centers of each bin 
        counts  : number of spikes included at each time point, in each category
        trials  : number of trials included at each time point, in each category
        rate : counts / trials
        columns : column header of counts and trials, that is, the category
            labels (could be neurons, stimuli, MultiIndex, etc)
    
    Class methods
        from_folded(combine_trials=False) :
        from_dict_of_folded :
        
    """
    def __init__(self, counts, trials, edges=None, t=None):
        # Convert to DataFrame (unless already is)
        self.counts = pandas.DataFrame(counts)
        self.trials = pandas.DataFrame(trials)
        self.columns = self.counts.columns
        
        # set up time points
        if t is None and edges is None:
            self.t = np.arange(counts.shape[0])
        elif t is None and edges is not None:
            self.t = edges[:-1] + np.diff(edges) / 2.
        else:
            self.edges = edges
            self.t = t
        
        # calculate rate
        self.rate = counts / trials.astype(np.float)
    
    @classmethod
    def from_folded(self, folded, starts=None, stops=None, durations=None,
        bins=None):
        """Given a list of locked spike-times, bin.

        Variables:
        folded : list-like, each entry is an array of locked times
        starts : list-like, same length as `folded`, the time at which 
            spike collection began for each trial
        stops, durations : when the spike collection ended for each trial

        TODO: issue warning if spikes occured outside of starts and stops

        Returns:
            Binned object containg counts and trials

        Recast this as a data transformation from
        * Representation 1: list of spike times, grouped by trial number
            One limitation of this representation is that there is no indication
            of the epoch over which the spike times might have come.
        * Representation 2: binned spike times, with trials in columns
            Thus, return a single DataFrame (or other sort of object) instead
            of the various ones returned here.
        
        Also write other methods that do the same transformation but via
        smoothing instead of binning.
        """
        # Iterate over the provided trials
        counts, edges = np.histogram(np.concatenate(folded), bins=bins)
        
        if stops is None:
            stops = starts + durations
        
        # count trials included in each bin
        trials = np.array([np.sum((stops - starts) > e) for e in edges[:-1]])

        return Binned(counts=counts, trials=trials, edges=edges)
    
    @classmethod
    def from_dict_of_folded(self, dfolded, starts=None, stops=None, durations=None,
        bins=None):
        """Given a list of locked spike-times, bin.

        Variables:
        dfolded : dict {trial_label : array of locked spike times}
        starts : dict {trial_label : start times of each trial}
            if None, then assume start and stop equal to bins
        stops : dict {trial_label : stop times of each trial}
            if None, then assume start and stop equal to bins

        Returns:
            counts, each trial is a column
            trials
            smoothed

        Recast this as a data transformation from
        * Representation 1: list of spike times, grouped by trial number
            One limitation of this representation is that there is no indication
            of the epoch over which the spike times might have come.
        * Representation 2: binned spike times, with trials in columns
            Thus, return a single DataFrame (or other sort of object) instead
            of the various ones returned here.
        
        Also write other methods that do the same transformation but via
        smoothing instead of binning.
        """
        # Create return values
        counts_l = []
        trials_l = []
        
        try:
            keys = dfolded.keys()
        except AttributeError:
            keys = range(len(dfolded))
        
        #~ binned = Binned(columns=keys)
        #~ for key in keys:
            #~ folded = dfolded[key]
            #~ binned[key] = from_folded(dfolded[key], starts=starts[key], 
                #~ stops=stops[key], bins=bins)
        
        # Iterate over the provided trials
        for n, key in enumerate(keys):
            folded = dfolded[key]
            
            # count spikes 
            counts, edges = np.histogram(np.concatenate(folded), 
                bins=bins)
            counts_l.append(counts)
            
            # how many trials were actually included
            if starts is None and stops is None:
                trials_l.append(np.ones(counts.shape[0], dtype=np.int) * len(folded))
            else:
                trials_l.append(np.array(
                    [np.sum(stops[key] - starts[key] > e) for e in edges[:-1]]))

        counts = pandas.DataFrame(data=np.asarray(counts_l).transpose(), 
            columns=keys)
        trials = pandas.DataFrame(data=np.asarray(trials_l).transpose(), 
            columns=keys)
        
        return Binned(counts=counts, trials=trials, edges=edges)

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
    

def timelock(a1, a2, start=0, stop=0, center=True):
    """Returns list of peri-event times.
    
    a1 : sorted array of times to lock
    a2 : set of events to lock on
    start : number, or array of shape a2, one for each event
    stop : number, or array of shape a2, one for each event
    center : center each list to its locking event
    
    Returns:
    List of same length as `a2`. Each entry is an array of times
    that occurred within [event - start, event + stop)
    """
    res = []
    i_starts = np.searchsorted(a1, a2 + start)
    i_stops = np.searchsorted(a1, a2 + stop)
    
    if center:
        for i_start, i_stop, aa2 in zip(i_starts, i_stops, a2):
            res.append(a1[i_start:i_stop] - aa2)
    else:
        for i_start, i_stop in zip(i_starts, i_stops):
            res.append(a1[i_start:i_stop])
    
    return res

def startswith(df, colname, s):
    # untested
    ixs = map(lambda ss: ss.startswith(s), df[colname])
    return df[ixs]


# Sandbox of methods to load data into canonical form
def get_events_and_trials(dirname, bskip=0):
    """Returns times of events and trials info.
    
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



# PLOTTING stuff
def plot_psth_with_rasters_from_dict(
    dfolded, keys=None, spshape=None, bins=None):
    
    if keys is None:
        keys = dfolded.keys()
    
    f = plt.figure()
    for n, key in enumerate(keys):
        folded = dfolded[key]
        ax = f.add_subplot(spshape[0], spshape[1], n + 1)        
        plot_psth_with_rasters(folded, bins=bins, ax=ax)
    
    plt.show()

def plot_psth_with_rasters(folded, smoothed=None, bins=None, ax=None):
    myutils.plot_rasters(folded, ax=ax)
    
    if smoothed is None:
        smoothed = Binned.from_folded(folded, bins=bins)
    ax.plot(smoothed.t, smoothed.rate)
    