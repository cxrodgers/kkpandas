"""Base objects for holding discrete events or spikes.

This module defines several ways of representing the times of events (or 
spikes), and when they occur time-locked to other events. Methods converting
between these representation are also provided.

The usual pipeline for spike time analysis is:
1) Select a set of spikes produced by one or more units
2) Select a set of events and define a time window of interest around them
3) Timelock the spikes to those events
4) Smooth or bin the spikes from each window

The following data structures implement those representations.

Flat - A flat representation is just an array of spike times or 
a DataFrame (with a column 'time') that has additional descriptive information
about each event. There is no "Flat" class ... anything that looks like that
will do.

Folded - A collection of Flat. Right now implemented as list-like, that is,
a list of Flat. Each entry in Folded is considered a "trial", a single Flat
representation of a certain epoch of time. The events in each trial can be
in the original time base, or locked to an event within each trial.

Binned - Smoothed across time and collapsing across replicates. This is a
PSTH-like object. Implemented as a DataFrame with time bins on the rows
and "categories of trials" on the columns. You define which trials are included
in each category .. in fact each category could be a single trial. The key
point is that all of the Flat representations have been turned into
continuously valued functions of time, by binning or smoothing for instance.
"""

import numpy as np
import pandas
from utility import timelock

class Folded:
    """Stores spike times on each trial timelocked to some event in that trial.

    Provides iteration over these spikes from each trial, as well as
    remembering the time base of each trial.
    
    TODO: allow the storing of trial labels so that trials can be indexed
    by label instead of index.
    
    TODO: be more flexible about time-locking being optional. Perhaps a flag
    so that it knows whether they are time-locked. Perhaps a way to easily
    convert between time-locked and original spike times.
    """
    def __init__(self, values, starts, stops, centers=None, 
        subtract_off_center=False, range=None, dataframe_like=None):
        """Initialize a new Folded.
        
        Generally you do not call this directly, but use one of the class
        method constructors (like `from_flat`).
        
        values : list or DataFrame of length n_trials.
            Each entry contains the spike times on that trial.
            Each entry is aligned to the corresponding entry in `centers`
        
        dataframe_like : Flag controlling whether the values are DataFrame
            or array. 
            If True, it should contain a column 'time', as well
            as potentially other columns (eg, 'unit', 'tetrode').
            If False, the values will be coerced to 1-dimensional arrays.
        
        subtract_off_center: If True, then subtract each entry in 'centers'
            from each entry in 'values', to time-lock to an event in the trial.
            If False, do not do this (for example if you have already
            time-locked your spikes, or time-locking is not meaningful.)
    
        These arrays are all of the same length:
        starts : array of start times by trial
            Currently this is a required argument.
            TODO: make optional, in case values are already time-locked.
        stops : array of stop times by trial
            TODO: see above.
        centers : array of trigger times by trial
            If not specified, uses starts
        
        range : A tuple (t_start, t_stop) for suggesting a range over which
        PSTH can be calculated.    
            If not specified, uses largest starting and stopping times
            over all trials.
            Oops ... using `range` as a variable name is a terrible idea.
            But this is what np.histogram uses ...
        """
        self.values = values
        self.starts = np.asarray(starts)
        self.stops = np.asarray(stops)
        
        # Guess whether dataframe like
        if dataframe_like is None:
            if len(values) > 0:
                try:
                    values[0]['time']
                    dataframe_like = True
                except (KeyError, ValueError):
                    dataframe_like = False
        self.dataframe_like = dataframe_like
        
        # Coerce values to 1-dim array
        # Otherwise we get weird errors from 0d entries
        # Add a test case for this
        if not self.dataframe_like:
            # Tortured syntax
            # Avoids conflict with keyword range
            # and also ensures that values[n] is a reference, not a copy
            for n in np.arange(len(values), dtype=np.int):
                values[n] = np.asarray(values[n]).flatten()
        
        # Store or calculate centers
        if centers is None:
            self.centers = starts
        else:
            self.centers = np.asarray(centers)
        
        # Store or calculate range
        if range is None:
            try:
                t_start = np.min(starts - centers)
                t_stop = np.max(stops - centers)
                self.range = (t_start, t_stop)
            except ValueError:
                self.range = None
        else:
            self.range = range
        
        # Optionally subtract off center
        if subtract_off_center:
            for val, center in zip(self.values, self.centers):
                if self.dataframe_like:
                    val['time'] -= center
                else:
                    val -= center
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __len__(self):
        return len(self.values)
    
    @classmethod
    def from_flat(self, flat, starts=None, centers=None, stops=None, 
        dstart=None, dstop=None, subtract_off_center=True):
        """Construct Folded from Flat.
        
        flat : A flat representation of spike times. It could be a simple
            array of times, or a DataFrame with a column 'time'.
        starts, centers, stops, dstart, dstop : ways of specifying trial
            windows. See `timelock`
        subtract_off_center : whether to align events to trigger on each
            trial
        """
        # Figure out whether input is structured or simple
        dataframe_like = True
        try:
            spike_times = flat['time']
        except (KeyError, ValueError):
            spike_times = flat
            dataframe_like = False
    
        # Get indexes into flat with timelock
        # We need to get the starts/centers/stops as actually calculated
        idx, starts, centers, stops = timelock(spike_times, 
            a2=centers, start=starts, stop=stops, dstart=dstart, dstop=dstop,
            return_value='index', error_check=True, return_boundaries=True)
        
        # Turn indexes back into values
        if dataframe_like:
            # Reconstruct values by indexing back into flat
            res = [flat.ix[flat.index[iix]] for iix in idx]
        else:
            # Simple input, could have used return_value='original' above
            res = [flat[iix] for iix in idx]
        
        # Construct Folded, using trial boundaries as calculate, and
        # subtracting off triggers
        return Folded(values=res, starts=starts, stops=stops, centers=centers,
            subtract_off_center=subtract_off_center)
    

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
    _FLOAT_EQ_ERR = 1e-7
    
    def __init__(self, counts, trials, columns=None, edges=None, t=None):
        """Prefer initialization with edges, but not t"""
        # Convert to DataFrame (unless already is)
        self.counts = pandas.DataFrame(counts)
        self.trials = pandas.DataFrame(trials)
        
        # Initialize category names
        if columns is not None:
            if self.counts.columns is None:
                # Actually this never occurs
                # Is this a use-case?
                self.counts.columns = columns
            else:
                self.counts = self.counts[columns]
            
            if self.trials.columns is None:
                self.trials.columns = columns
            else:
                self.trials = self.trials[columns]
            
        
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
        #self.rate = counts / trials.astype(np.float)
    
    # Wrapper functions around DataFrame methods
    def __getitem__(self, key):
        """This actually needs to work for either time indexes or 
        column names indexes"""
        return Binned(counts=self.counts[key], trials=self.trials[key],
            t=self.t[key])
    
    def rename(self, columns=None, multi_index=True):
        """Inplace rename columns"""
        self.counts = self.counts.rename(columns=columns)
        self.trials = self.trials.rename(columns=columns)
        
        if multi_index:
            self.tuple2multi()
    
    def reorder(self, columns=None):
        """Inplace reorder columns"""
        self.counts = self.counts[columns]
        self.trials = self.trials[columns]
    
    def drop(self, labels, axis=1):
        """Note default behavior is to drop column.
        
        Also deals with string column labels
        """
        if not is_nonstring_iter(labels):
            labels = [labels]
        self.counts = self.counts.drop(labels, axis)
        self.trials = self.trials.drop(labels, axis)
    
    def sum(self, axis=0, level=None):
        return Binned(
            counts=self.counts.sum(axis=axis, level=level),
            trials=self.trials.sum(axis=axis, level=level),
            edges=self.edges)
   
    @property
    def columns(self):
        assert np.all(
            self.counts.columns.values == self.trials.columns.values)
        return self.counts.columns
    
    # Convenience methods for things I can never remember how to do
    def tuple2multi(self):
        if is_nonstring_iter(self.columns[0]):
            idx = pandas.MultiIndex.from_tuples(self.columns)
            self.counts.columns = idx
            self.trials.columns = idx

    def sum_columns(self, level=None):
        """Sum over categories. Specify level if multi-indexing"""
        return self.sum(axis=1, level=level)
    
    def sum_rows(self):
        return Binned(
            counts=self.counts.sum(axis=0),
            trials=self.trials.mean(axis=0))

    # Rate calculation
    @property
    def rate(self):
        return self.counts / self.trials.astype(np.float)
    
    # Construction methods    
    @classmethod
    def from_folded(self, folded, bins=None, starts=None, stops=None):
        """Construct Binned object by histogramming list-like Folded.
        
        It is assumed that folded contains replicates to be averaged together.
        Thus, this returns a Binned with one category. 
        
        If you are trying to form a Binned with more than one category, see
        from_dict_of_folded
        
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
            try:
                times = np.concatenate(folded)
            except ValueError:
                # all empty?
                times = np.array([])

        # Here is the actual histogramming
        counts, edges = np.histogram(times, bins=bins, range=range)
        
        # Now we calculate how many trials are included in each bin
        trials = np.array([np.sum((stops - starts) > e) for e in edges[:-1]])
        
        # Now construct and return
        return Binned(counts=counts, trials=trials, edges=edges)
    
    @classmethod
    def from_dict_of_folded(self, dfolded, keys=None, bins=100, binwidth=None):
        """Initialize a Binned from a dict of Folded over various categories
        
        Simple wrapper function that creates a Binned from each Folded,
        then sticks them together into a single Binned.
        
        dfolded : dict from category name to Folded of replicates from
            that category
        keys : ordered category names that you wish to include. If None,
            uses dfolded.keys()
        bins : number or array
            If you wish to specify bins exactly, pass as array
            If you specify the number of bins, then in order to keep the
            time base consistent, the `range` attribute of each folded
            is queried and the smallest interval covering all ranges is used.
        binwidth: width of bin in seconds
            If you specify bins as a number and binwidth, binwidth dominates.
        """
        if keys is None:
            keys = dfolded.keys()
        
        # Auto set the bins
        if not np.iterable(bins):
            # Determine spanning range
            all_ranges = np.array([val.range for val in dfolded.values()])
            range = (all_ranges[:, 0].min(), all_ranges[:, 1].max())
        
            # Set via width or number
            if binwidth is None:
                bins = np.linspace(range[0], range[1], bins)
            else:
                bins = np.arange(range[0], range[1], binwidth)
        
        binned_d = {}
        for key in keys:
            binned_d[key] = Binned.from_folded(dfolded[key], bins=bins)

        return Binned.from_dict_of_binned(binned_d, keys=keys)
    
    @classmethod
    def from_dict_of_binned(self, dbinned, keys=None):
        """Initialize a Binned from a dict of Binned.
        
        This is a concatenation-like operation: the result contains
        each of the values in dbinned in columns titled by keys
        
        TODO: make this work when the keys of dbinned are tuples. This should
        probable generate a multi-index on the columns
        """
        # If no keys specified, use all keys in sorted order
        if keys is None:
            keys = sorted(dbinned.keys())

        # Construct counts and trials by concatenating the underlying
        # objects. This method actually results in a MultiIndex with the
        # first level being `key`. We override below, though perhaps
        # this is actually a more reasonable behavior ...
        # Note this also randomizes the column order, but we'll define
        # it in the call to Binned
        all_counts = pandas.concat(
            {key: dbinned[key].counts for key in keys}, axis=1)
        all_counts.columns = [c[0] for c in all_counts.columns]
        all_trials = pandas.concat(
            {key: dbinned[key].trials for key in keys}, axis=1)
        all_trials.columns = [c[0] for c in all_trials.columns]    
        
        # The time base should be the same
        all_edges = np.array([dbinned[key].edges for key in keys])
        edges = np.mean(all_edges, axis=0)
        err = np.max(np.abs(all_edges - edges))
        if err > self._FLOAT_EQ_ERR:
            raise ValueError("dict of binned appear not to share timebase")
        
        # Construct (note override of column names)
        return Binned(counts=all_counts, trials=all_trials, edges=edges,
            columns=keys)
