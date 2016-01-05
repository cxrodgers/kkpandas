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
import copy, warnings

class Folded:
    """Stores spike times on each trial timelocked to some event in that trial.

    Provides iteration over these spikes from each trial, as well as
    remembering the time base of each trial.
    
    TODO: be more flexible about time-locking being optional. Perhaps a flag
    so that it knows whether they are time-locked. Perhaps a way to easily
    convert between time-locked and original spike times.
    """
    def __init__(self, values, starts, stops, centers=None, labels=None,
        subtract_off_center=False, range=None, dataframe_like=None):
        """Initialize a new Folded.
        
        Generally you do not call this directly, but use one of the class
        method constructors (like `from_flat`).
        
        values : list or DataFrame of length n_trials.
            Each entry contains the spike times on that trial.
            Each entry is aligned to the corresponding entry in `centers`
            
            values HAS to be stored internally as a list.
            if it is an array, then the dtype gets messed up for empty arrays
        
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
        labels : array of labels by trial
            Generally (always?) an integer representing the trial number
        
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
                except (KeyError, ValueError, TypeError, IndexError):
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
        
        # Store labels if provided
        if labels is None:
            self.labels = None
        else:
            self.labels = np.asarray(labels)
        
        # Store or calculate range
        if range is None:
            try:
                t_start = np.min(self.starts - self.centers)
                t_stop = np.max(self.stops - self.centers)
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
    
    def get_slice(self, slc):
        """Returns a Folded with just the values that are True in mask
        
        Converts my data into arrays to slice with `slc`, then returns
        new object.
        
        This could go in __getitem__ but I'm afraid that would lead to weird
        bugs.
        
        So, use __getitem__ for simple indexing: f[0], f[2:5]
        Use this for fancy indexing: 
            f.get_slice([0, 2, 4]), f.get_slice(f.apply(len) > 1)
        """
        slc = np.asarray(slc) # in case a list was passed
        
        if len(slc) == 0:
            return Folded(starts=[], centers=[], stops=[], values=[],
                range=self.range, labels=[])
        
        # Convert to array to do the slicing
        # Most will be converted to array in constructor, except for values
        # So leave values as a list
        if self.starts is None:
            starts = None
        else:
            starts = np.asarray(self.starts)[slc]

        if self.values is None:
            values = None
        else:
            # something gets all messed up here with fancy indexing
            # if all of the entries in self.values are empty
            if slc.dtype == np.dtype('bool'):
                values = [self.values[i] for i in np.where(slc)[0]]
            else:
                values = [self.values[i] for i in slc]

        if self.stops is None:
            stops = None
        else:
            stops = np.asarray(self.stops)[slc]

        if self.centers is None:
            centers = None
        else:
            centers = np.asarray(self.centers)[slc]

        if self.labels is None:
            labels = None
        else:
            labels = np.asarray(self.labels)[slc]

       
        # Construct the return value
        ret = Folded(starts=starts, centers=centers, stops=stops, values=values,
            range=self.range, labels=labels)
        
        return ret
    
    def __getitem__(self, key):
        try:
            ret = self.values[key]
        except TypeError:
            raise TypeError("cannot index by %r, try get_slice" % key)
        return ret
    
    def __len__(self):
        return len(self.values)
    
    def __repr__(self):
        l = len(self)
        ret = "Folded containing %d spiketrains\n" % l
        if l > 0:
            ret += "%r\n" % self.values[0]
        if l > 1:
            ret += "...\n%r" % self.values[-1]
        return ret
    
    def __add__(self, other):
        """Add two Folded by concatenating their values"""
        # Concatenate all by converting to list first
        # Most will be converted to array in constructor, except for values
        try:
            starts = list(self.starts) + list(other.starts)
        except:
            starts = None
        try:
            values = list(self.values) + list(other.values)
        except:
            values = None
        try:
            stops = list(self.stops) + list(other.stops)
        except:
            stops = None
        try:
            centers = list(self.centers) + list(other.centers)
        except:
            centers = None  
        try:
            labels = list(self.labels) + list(other.labels)
        except:
            labels = None

        # Check that ranges are consistent, if possible
        try:
            if not np.allclose(self.range, other.range):
                print "warning: range not the same in summed Folded"
        except:
            pass
        
        # Take the mean of the ranges, if possible
        try:
            range = np.mean([self.range, other.range], axis=0)
        except:
            range = None
        
        # Construct the return value
        ret = Folded(starts=starts, centers=centers, stops=stops, values=values,
            range=range, labels=labels)
        
        return ret
    
    def apply(self, func):
        """Applies func to each spiketrain in this Folded.
        
        Returns the results as an array.
        """
        return np.asarray(map(func, self.values))
    
    def count_in_window(self, start, stop):
        """Counts spikes in time window on each spiketrain.
        
        Return the number of spike times greater than `start` and less than
        `stop` for each spiketrain.
        """
        func = lambda st: sum((st >= start) & (st < stop))
        return self.apply(func)
    
    @classmethod
    def from_flat(self, flat, starts=None, centers=None, stops=None,
        dstart=None, dstop=None, subtract_off_center=True, range=None,
        labels=None):
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
        except (KeyError, ValueError, TypeError, IndexError):
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
            subtract_off_center=subtract_off_center, range=range, labels=labels)


# Function to compare two folded
# Should maybe be an __eq__ method of folded?
def what_differs(folded1, folded2):
    """Returns a string about what differs, or None if equal"""
    if len(folded1) != len(folded2):
        return 'different lengths'
    
    if not np.allclose(folded1.starts, folded2.starts):
        return 'starts'
    if not np.allclose(folded1.stops, folded2.stops):
        return 'stops'
    if not np.allclose(folded1.centers, folded2.centers):
        return 'centers'
    if not np.allclose(folded1.range, folded2.range):
        return 'range'
    
    # Test labels
    if folded1.labels is None:
        if folded2.labels is None:
            pass
        else:
            return 'labels case1'
    else:
        if folded2.labels is None:
            return 'labels case2'
        else:
            if not np.allclose(folded1.labels, folded2.labels):
                return 'labels case3'

    # Test values
    for val1, val2 in zip(folded1.values, folded2.values):
        if not np.allclose(val1, val2):
            return 'values'
    return None

def is_equal(folded1, folded2):
    """Returns True if folded1 and folded2 are equal, up to floating point"""
    return what_differs(folded1, folded2) is None

    

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
        """Initialize a new Binned.
        
        The preferred way to initialize is by specifying `edges`, not `t`,
        because it easier to derive `t` from `edges` than vice versa.
        
        Also note that `edges` is best calculated using linspace, not arange,
        to mitigate floating point error.
        """
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
    
    def __len__(self):
        return len(self.counts)
    
    @property
    def shape(self):
        return self.counts.shape
    
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
    
    def rate_in(self, units='Hz'):
        rate = self.rate
        
        if units == 'Hz' or units == 'hz':            
            dt = np.diff(self.edges).mean()
            rate = rate / dt
        elif units is None:
            pass
        else:
            raise ValueError("unknown unit %s" % units)
        return rate
    
    # Construction methods    
    @classmethod
    def from_folded(self, folded, bins=None, starts=None, stops=None,
        meth=np.histogram):
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
        meth : a method, default np.histogram, that takes a concatenated list
            of spike times and produces a "rate over time". It will also
            receive the `range` attribute of `folded`, if available.
            Another option:
            gs = kkpandas.base.GaussianSmoother(smoothing_window=.005)
            meth = gs.smooth
        
        The following attributes are collected from `folded` if available,
        or otherwise you can specify them as arguments. They are necessary
        because this method needs to know when spike collection began and
        ended for each trial.
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
        counts, edges = meth(times, bins=bins, range=range)
        
        # Now we calculate how many trials are included in each bin
        trials = np.array([np.sum((stops - starts) > e) for e in edges[:-1]])
        
        # Now construct and return
        return Binned(counts=counts, trials=trials, edges=edges)
    
    @classmethod
    def from_dict_of_folded(self, dfolded, keys=None, bins=100, binwidth=None,
        **kwargs):
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
        kwargs : sent to from_folded for each category
        
        TODO: harmonize the generation of `bins` with from_folded
        Right now this one is different, more feature-ful, but also fails
        if the range attribute is not set on the values in dfolded.
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
            binned_d[key] = Binned.from_folded(dfolded[key], bins=bins, **kwargs)

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
    
    @classmethod
    def from_folded_by_trial(self, folded, bins=None, starts=None, stops=None,
        range=None, **kwargs):
        """Bin each trial separately.
        
        Right now this is a little hacky. It just creates a dict from 
        each trial index to each trial in folded, then calls
        from_dict_of_folded
        
        starts, stops, range : used for folding each trial, otherwise
            taken from folded
        bins, **kwargs : sent to Binned constructor
        
        A common kwarg is `meth`, which is np.histogram by default, but
        could be a smoother.
            gs = kkpandas.base.GaussianSmoother(smoothing_window=.005)
            meth = gs.smooth
        """
        # Copy data into individual foldeds
        # This should probably be a method in Folded
        if starts is None:
            starts = folded.starts
        if stops is None:
            stops = folded.stops
        if range is None:
            range = folded.range
        
        dfolded = {}
        for n in np.arange(len(folded), dtype=np.int):
            # Create a folded from a single trial
            # Would be nice if this were a method in folded
            ff = Folded(values=[folded[n]],
                starts=[starts[n]], stops=[stops[n]], range=range,
                subtract_off_center=False) 
            dfolded[n] = ff

        return Binned.from_dict_of_folded(dfolded, bins=bins, **kwargs)

def folded_rate(folded):
    """Utility funtion for rate by trial in folded. Put this in folded."""
    res = []
    for t1, t2, vl in zip(folded.starts, folded.stops, folded.values):
        res.append(len(vl) / float(t2 - t1))
    return np.asarray(res)

def concat_binned(list_of_binned, keys):
    """TODO. See Binnned.from_dict_of_binned"""
    pass

def define_bin_edges(bins=None, binwidth=None, range=None):
    """Determine bin edges given width or number, and a range to span.
    
    Specifying the number of bins is more resistant to floating point error
    than specifying the width of the bin.
    
    bins : number, or array-like
    binwidth : specified width of each bin
    range : start, stop
    
    Returns: edges
    """
    if np.iterable(bins):
        edges = bins
    else:
        if range is None:
            raise Exception("you must specify the range")
        
        # Set via width or number
        if binwidth is None:
            edges = np.linspace(range[0], range[1], bins + 1)
        else:
            # TODO: change this to bins = np.rint(np.diff(range) / binwidth)
            edges = np.arange(range[0], range[1], binwidth)    
    
    return edges


def define_range(data_range=None, t_start=None, t_stop=None, times=None,
    range=None):
    """Defines the range of a set of times.
    
    1.  If `data_range` is a 2-tuple, we start with this for the data range.
        Otherwise we start with (None, None)
    2.  Replace any None in the 2-tuple with t_start, t_stop
    3.  Replace any remaining None with times.min(), times.max()
    
    If none of these strategies work, raises ValueError rather than return
    a munged data range.
    
    `range` is an obsolete synonym for `data_range` that will generate
    a warning.
    """
    # Obsolete keyword `range`
    if range is not None:
        warnings.warn('`range` is an obsolete keyword arg, use `data_range`')
        if data_range is None:
            data_range = range
    
    # First listify
    if data_range is None:
        data_range = [None, None]
    else:
        data_range = list(data_range)

    # Start with t_start, t_stop if no full range provided
    if data_range[0] is None:
        data_range[0] = t_start
    if data_range[1] is None:
        data_range[1] = t_stop
    
    # Replace with data actual range
    try:
        if data_range[0] is None:
            data_range[0] = np.min(times)
        if data_range[1] is None:
            data_range[1] = np.max(times)
    except ValueError:
        raise ValueError(
            "cannot take min or max of %r to calculate range" % times)
    
    # Error check
    if None in data_range:
        raise ValueError("cannot calculate data range from " +
            "data_range: %r t_start: %r t_stop: %r times: %r" % (
            data_range, t_start, t_stop, times))
    
    return tuple(data_range)

def define_bin_edges2(bins=None, binwidth=None, data_range=None, 
    t_start=None, t_stop=None, times=None, range=None):
    """Canonical wawy to determine bin edges from various desiderata.
    
    This should be phased in in place of define_bin_edges.
    
    If `bins` is already an iterable, it is returned without further ado.
    Otherwise, the data range is calculated using `define_range` and the
    following kwargs: `data_range`, `t_start`, `t_stop`, `times`, `range`
    Note that `range` is obsolete!
    
    Then, if `bins` is an integer, it is used to specify the bin edges
    using `linspace` to span the data range. This is the preferred way.
    
    Otherwise, if `bins` is None, then `binwidth` is used to calculate a
    number of bins. This is more ambiguous, because if the data range is not
    equally divided by bin width, should the range or the binwidth be changed?
    This function changes to bin width to preserve the range. Note that
    rounding error can cause unpredictable effects here.
    
    Returns: bin_edges, with length 1 greater than the number of bins
    """
    # Stop if done
    if np.iterable(bins):
        if None in bins:
            warnings.warn('None in the provided bins, overwriting')
        else:
            return bins
    
    # Calculate range
    data_range = define_range(data_range=data_range, t_start=t_start,
        t_stop=t_stop, times=times, range=range)
    
    # Error check
    # We could return a backwards range but is this ever useful?
    if data_range[0] > data_range[1]:
        raise ValueError("I got a backwards range: %d" % data_range)
    
    # Use binwidth to estimate number of bins if necessary
    if bins is None:
        bins = np.rint((data_range[1] - data_range[0]) / binwidth)
    
    # Calculate
    edges = np.linspace(data_range[0], data_range[1], bins + 1)  
    return edges

from DiscreteAnalyze.PointProc import smooth_event_train
class GaussianSmoother:
    """Object providing methods to smooth spiketrains instead of binning.
    
    Primarily intended to provide the `meth` argument to Binned.from_folded
    
    Rather than filter using filtfilt, this adds Gaussians at each timestamp,
    which is probably more efficient for most spiketrains.
    """
    def __init__(self, smoothing_window=.005):
        self.smoothing_window = smoothing_window

    def smooth(self, a, bins=None, range=None):
        """Add Gaussians at each time in `times`.
        
        Intended as drop-in replacement for np.histogram.
        The continously valued smooth signal is sampled at each of the
        bin centers.
        
        The variable bins determines the precision of the returned signals.
        The variable smoothing_window determines the smeared-outness of
        the gaussians.
        
        Because the spike times are discretized first, all the Gaussians
        will have the same peak (that is, no penalty for being slightly off
        a bin center). This is not as accurate but for bin-spacing much
        finer than the smoothing_window, it is okay.
        
        bins - instead of bin edges, this is more like sampling points
            of the continuous valued smoothed signal. This should be
            faster than self.smoothing_window because the spikes are discretized
            and spread out with stdev smoothing_window.
        range - used to define bin edges if bins is an integer
        
        Example:
            bins = [-.002, 0., .002, .004]
            spike_times = [-.0005, .0005]
            bincenters = [-.001, .001, .003]
            f_samp = 1/.002 = 500
            t_start, t_stop = -.001, .003
            
            We discretize by multiplying by f_samp and rounding:
            n_start, n_stop = 0, 2
            timestamps = [0, 0]
            
            The returned values are n_op = [0, 1, 2] and x_op evaluated at
            those points. In ths case that will be a half Gaussian with peak
            at 0.
        
        Gain:
            Currently, the underlying smoothing function adds a Gaussian
            that has been normalized in the discretized basis. That is,
            the returned signal will always sum to len(a) by construction,
            unless some spikes have been chopped off due to truncation.
            You could argue that this should be in Hz, or that the normalization
            of each Gaussian should not account for the amount lost to
            the approximation inherent in discretization.
        
        Returns:
            counts, edges
            counts - value at each center in between edges
            edges - the bin edges
        """
        # Define bins and sampling frequency
        bins = np.asarray(define_bin_edges(bins=bins, range=range))
        bincenters = bins[:-1] + 0.5 * np.diff(bins)
        binwidth = np.mean(np.diff(bincenters))
        fs = 1 / binwidth

        # Discretize by numbering the bins from 0 to len(bincenters) - 1,
        # and subtracting the first bincenter from all spike times, then
        # multiplying by sampling rate.
        n_start = 0
        n_stop = len(bincenters) - 1
        timestamps = np.rint(fs*(np.asarray(a) - bincenters[0])).astype(np.int)
        
        # Define filter parameters in samples
        filter_std = self.smoothing_window * fs # ok if not integer
        filter_truncation_width = np.rint(3 * filter_std).astype(np.int)
        
        # Call the underlying smoothing function
        n_op, x_op = smooth_event_train(
            timestamps=timestamps, filter_std=filter_std,
            filter_truncation_width=filter_truncation_width, 
            n_min=n_start, n_max=n_stop)
        
        # Cast into desired units
        counts = x_op
        edges = bins

        return counts, edges