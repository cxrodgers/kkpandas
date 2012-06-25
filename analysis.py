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


The main representations are as follows:
* Flat : list of spike times associated with one or more units
* Folded : list of flat representations, one for each trial
* Smoothed : table of histogrammed spike times, one for each category
  (which could be a single trial)

Computation objects
* TimePicker : generates a list of times to lock on, for instance trials of
 a certain type
* UnitPicker : generates a list of units to analyze
These are used to choose times / units to fold on


Define these objects here. Then, another module for derived objects that
do common operations for my data, eg, choosing stimuli of a certain type.


"""



import pandas
import numpy as np
from ns5_process import bcontrol
import os.path

class Folded:
    """Stores spike times on each trial timelocked to some event in that trial.

    Provides iteration over these spikes from each trial, as well as
    remembering the time base of each trial.
    """
    def __init__(self, values, starts, stops, centers=None, range=None):
        """Initialize a new Folded
        
        values : list of times on each trial, also accessible with getitem
        (though perhaps it should be some kind of trial-label?)
        Each entry is aligned to the corresponding entry in `centers`
    
        These arrays are all of the same length:
        starts : array of start times by trial
        stops : array of stop times by trial
        centers : array of trigger times by trial
            If not specified, uses starts
        
        range : A tuple (t_start, t_stop) for suggesting a range over which
        PSTH can be calculated.    
            If not specified, uses largest starting and stopping times
            over all trials.
        """
        self.values = values
        self.starts = starts
        self.stops = stops
        
        if centers is None:
            self.centers = starts
        else:
            self.centers = centers
        
        if range is None:
            t_start = np.min(starts - centers)
            t_stop = np.max(stops - centers)
            self.range = (t_start, t_stop)
        else:
            self.range = range
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __len__(self):
        return len(self.values)
    
    @classmethod
    def from_flat(self, flat, starts, centers=None, stops=None, durations=None):
        """Construct Folded from Flat.
        
        flat : representation of events with column `time`
        starts : beginning of each trial (ie, entry in returned result)
        stops : end of each trial
        
        If stops is None, adds `durations` to starts and uses that.
        If `durations` is also None, then uses all of the data (by using
        the next start as the stop).
        """
        starts = np.asarray(starts)
        
        if centers is None:
            centers = starts

        # Define stops somehow
        if stops is None:
            if durations is not None:
                stops = starts + durations
            else:
                stops = list(starts[1:])
                stops.append(flat.time.max())
                stops = np.asarray(stops)
        
        # Extract and append
        res = []
        for start, stop, center in zip(starts, stops, centers):
            res.append(flat[(flat.time >= start) & (flat.time < stop)])
            res[-1]['time'] -= center

        # Construct and return
        return Folded(values=res, starts=starts, stops=stops, centers=centers)
    

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
    def __init__(self, counts, trials, columns=None, edges=None, t=None):
        """Prefer initialization with edges, but not t"""
        # Convert to DataFrame (unless already is)
        self.counts = pandas.DataFrame(counts)
        self.trials = pandas.DataFrame(trials)
        
        # Initialize category names
        if columns is None:
            self.columns = self.counts.columns
            assert np.all(
                self.counts.columns.values == self.trials.columns.values)
        else:
            self.columns = columns
            self.counts.columns = columns
            self.trials.columns = columns
        
        # set up time points
        if t is None and edges is None:
            # Nothing provided ... guess
            self.t = np.arange(counts.shape[0])
            self.edges = np.arange(counts.shape[0] + 1)
        elif t is None and edges is not None:
            # edges provided, calculate t
            self.t = edges[:-1] + np.diff(edges) / 2.
            self.edges = edges
        else:
            # only t provided, or both are provided
            self.edges = edges
            self.t = t
        
        # calculate rate
        self.rate = counts / trials.astype(np.float)
    
    @classmethod
    def from_folded(self, folded, bins=None, starts=None, stops=None):
        """Construct Binned object by histogramming list-like Folded.
        
        This returns a Binned with one category. To return multiple categories,
        see from_dict_of_folded
        
        Variables:
        folded : list-like, each entry is an array of locked times
        bins : passed to np.histogram. We also pass the `range` attribute
            of `folded` to histogram. This means that you can specify bins
            exactly (in which case `range` is ignored), or you can specify
            a number of bins (in which case `range` is used to ensure
            consistent bin sizes regardless of when spikes occurred). This
            assumes the `range` attribute of `folded` is specified correctly...
        
        The following attributes are collected from `folded` if available,
        or otherwise you can specify them as arguments.
        starts : list-like, same length as `folded`, the time at which 
            spike collection began for each trial
        stops : when the spike collection ended for each trial
        
        The purpose of these attributes is to construct the attribute
        `trials`, which contains the number of trials in each bin.
        """
        # Get trial times from object if necessary
        if starts is None:
            starts = np.asarray(folded.starts)
        
        if stops is None:
            stops = np.asarray(folded.stops)
        
        # Determine range
        try:
            range = folded.range
        except AttributeError:
            range = None
        
        # Put all the trials together (try to make it work for lists too)
        try:
            cc = pandas.concat(folded)
            times = cc.time
        except:
            times = np.concatenate(folded)

        # Here is the actual histogramming
        counts, edges = np.histogram(cc.time, bins=bins, range=range)
        
        # Now we calculate how many trials are included in each bin
        trials = np.array([np.sum((stops - starts) > e) for e in edges[:-1]])
        
        # Now construct and return
        return Binned(counts=counts, trials=trials, edges=edges)
    
    @classmethod
    def from_dict_of_folded(self, dfolded, keys=None, bins=None):
        """Initialize a Binned from a dict of Folded over various categories
        
        The category labels will be the keys to dfolded.
        """
        if keys is None:
            keys = dfolded.keys()
        
        binned_d = {}
        for key in keys:
            binned_d[key] = Binned.from_folded(dfolded[key], bins=bins)

        return Binned.from_dict_of_binned(binned_d, keys=keys)
    
    @classmethod
    def from_dict_of_binned(self, dbinned, keys=None):
        """Initialize a Binned from a dict of Binned."""
        # If no keys specified, use all keys in sorted order
        if keys is None:
            keys = sorted(dbinned.keys())

        # Construct counts and trials by concatenating the underlying
        # objects. This method actually results in a MultiIndex with the
        # first level being `key`. We override below, though perhaps
        # this is actually a more reasonable behavior ...
        all_counts = pandas.concat(
            {key: dbinned[key].counts for key in keys}, axis=1)
        all_trials = pandas.concat(
            {key: dbinned[key].trials for key in keys}, axis=1)
        
        # The time base should be the same
        all_edges = np.array([dbinned[key].edges for key in keys])
        edges = all_edges[0]
        for edges1 in all_edges:
            assert np.all(edges1 == edges)
        
        # Construct (note override of column names)
        return Binned(counts=all_counts, trials=all_trials, edges=edges,
            columns=keys)
    
    


    

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





# Utility functions for data frames
def startswith(df, colname, s):
    # untested
    ixs = map(lambda ss: ss.startswith(s), df[colname])
    return df[ixs]

def is_nonstring_iter(val):
    return hasattr(val, '__len__') and not isinstance(val, str)

def panda_pick(df, isnotnull=None, **kwargs):
    """Underlying picking function
    
    Returns indexes into trials_info for which the following is true
    for all kwargs:
        * if val is list-like, trials_info[key] is in val
        * if val is not list-like, trials_info[key] == val
    
    list-like is defined by responding to __len__ but NOT being a string
    (since this is commonly something we test against)
    
    isnotnull : if not None, then can be a key or list of keys that should
        be checked for NaN using pandas.isnull
    
    TODO:
    add flags for string behavior, AND/OR behavior, error if item not found,
    return unique, ....
    """
    msk = np.ones(len(df), dtype=np.bool)
    for key, val in kwargs.items():
        if is_nonstring_iter(val):
            msk &= df[key].isin(val)
        else:
            msk &= (df[key] == val)
    
    if isnotnull is not None:
        # Edge case
        if not is_nonstring_iter(isnotnull):
            isnotnull = [isnotnull]
        
        # Filter by not null
        for key in isnotnull:
            msk &= -pandas.isnull(df[key])

    return df.index[msk]

def panda_pick_data(df, **kwargs):
    """Returns sliced DataFrame based on indexes from panda_pick"""
    return df.ix[panda_pick(df, **kwargs)]





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
    
