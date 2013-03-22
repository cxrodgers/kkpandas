"""Utility methods to construct and operate on kkpandas objects.

This module includes low-level, efficient methods that are used in various
other objects.
* timelock : Folding operation
* panda_pick : Selection from DataFrame based on items in columns
"""



import pandas
import numpy as np
import os.path
    

def timelock(a1, a2=None, start=None, stop=None, dstart=None, dstop=None,
    return_value='original', error_check=True, return_boundaries=False,
    warn_if_overlap=True):
    """Returns list of peri-event times.
    
    This is the inner-loop in most spike analysis and is optimized here
    with np.searchsorted. That means you must pre-sort all arguments. Set
    `error_check` to False to disable run-time checking of this, which may
    improve speed by a small amount.
    
    a1 : sorted array of times to lock. Result consists of values from this.
    
    There are two ways to specify the "triggers", one for each event.
    Method 1:
        Specify `a2` as a sorted array of triggers.
        In this case you must also specify the window around each event, by
        specifying one of the following:
        `dstart` : Added to `a2` to calculate start of each trial, so can be
            array-like or number.
        `start` : Exact trial starts, so should be same shape as `a2`.
        
        You define `stop` or `dstop` similarly.
    Method 2:
        Let `a2` be None, and specify `start` as an array of trial starts.
        In this case, you must specify `stop` or `dstop` so that trial stops
        can be calculated by adding to `start`.
    Method 3:
        You only specify `a2` or `start` and nothing else. In this case
        these are used as the start/trigger times, and the stop times are
        the subsequent start time.
    
    You can specify overlapping trial windows but a warning is printed if
    `warn_if_overlap` is True.
    
    Returns:
    List of length equal to number of triggers. Each entry is an array of times
    from `a2` that occurred within a half-open interval around the triggers.
    
    The return value is always of the same shape, but takes one of the following
    values depending on `return_value`:
        'original' : the original times from `a1`
        'recentered' : the original times from `a1`, aligned to the triggers.
            That is, the trigger times are subtracted off. If Method 2 was
            used (see above), then the subtract off the start times.
        'index' : Indexes into `a1`
    
    If return_boundaries is True:
        will also return starts, centers, stops
    """
    a1 = np.asarray(a1)
    
    # Define a2 and start
    if a2 is None:
        # Method 2: Define center using start
        start = np.asarray(start)
        a2 = start
    else:
        # Method 1: Define start using center
        if start is None:
            if dstart is None:
                start = np.asarray(a2)
            else:
                start = np.asarray(a2) + np.asarray(dstart)
        else:
            start = np.asarray(start)

    # Define stop
    if stop is None:
        if dstop is not None:
            stop = np.asarray(a2) + np.asarray(dstop)
        else:
            # Method 3: greedy definition of stop
            # Each trial grabs up to the next start
            # The last trial goes up to infinity
            stop = np.concatenate([start[1:], [np.inf]])
    
    # Now some error checking
    if error_check:
        if np.any(a1 != np.sort(a1)):
            raise Exception("times must be sorted")
        if np.any(start != np.sort(start)):
            raise Exception("starts must be sorted")
        if np.any(stop != np.sort(stop)):
            raise Exception("stops must be sorted")
        if np.any(stop < start):
            raise Exception("stops must be after starts")
    
    if warn_if_overlap:
        if np.any(start[1:] < stop[:-1]):
            print "warning: trial overlap in timelock, possible doublecounting"
    
    # Find indexes into a1 using start and stop
    i_starts = np.searchsorted(a1, start)
    i_stops = np.searchsorted(a1, stop)
    
    # Form result list using those indexes
    res = []
    if return_value == 'original':
        for i_start, i_stop in zip(i_starts, i_stops):
            res.append(a1[i_start:i_stop])        
    elif return_value == 'recentered':
        for i_start, i_stop, aa2 in zip(i_starts, i_stops, a2):
            res.append(a1[i_start:i_stop] - aa2)
    elif return_value == 'index':
        for i_start, i_stop, aa2 in zip(i_starts, i_stops, a2):
            res.append(range(i_start, i_stop))
    else:
        raise Exception("unsupported return value: %s" % return_value)
    
    if return_boundaries:
        return res, start, a2, stop
    else:
        return res




# Utility functions for data frames
def startswith(df, colname, s):
    # untested
    ixs = map(lambda ss: ss.startswith(s), df[colname])
    return df[ixs]

def is_nonstring_iter(val):
    return hasattr(val, '__len__') and not isinstance(val, str)

def panda_pick(df, isnotnull=None, **kwargs):
    """Function to pick row indices from DataFrame.
    
    This method provides a nicer interface to choose rows from a DataFrame
    that satisfy specified constraints on the columns.
    
    isnotnull : column name, or list of column names, that should not be null.
        See pandas.isnull for a defintion of null
    
    All additional kwargs are interpreted as {column_name: acceptable_values}.
    For each column_name, acceptable_values in kwargs.items():
        The returned indices into column_name must contain one of the items
        in acceptable_values.
    
    If acceptable_values is None, then that test is skipped.
        Note that this means there is currently no way to select rows that
        ARE none in some column.
    
    If acceptable_values is a single string or value (instead of a list), 
    then the returned rows must contain that single string or value.
    
    TODO:
    add flags for string behavior, AND/OR behavior, error if item not found,
    return unique, ....
    """
    msk = np.ones(len(df), dtype=np.bool)
    for key, val in kwargs.items():
        if val is None:
            continue
        elif is_nonstring_iter(val):
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

def correlogram(t1, t2=None, bin_width=.001, limit=.02, auto=False):
    """Return crosscorrelogram of two spike trains.
    
    Essentially, this algorithm subtracts each spike time in `t1` 
    from all of `t2` and bins the results with numpy.histogram, though
    several tweaks were made for efficiency.
    
    Arguments
    ---------
        t1 : first spiketrain, raw spike times in seconds.
        t2 : second spiketrain, raw spike times in seconds.
        bin_width : width of each bar in histogram in sec
        limit : positive and negative extent of histogram, in seconds
        auto : if True, then returns autocorrelogram of `t1` and in
            this case `t2` can be None.
    
    Returns
    -------
        (count, bins) : a tuple containing the bin edges (in seconds) and the
        count of spikes in each bin.

        `bins` is relative to `t1`. That is, if `t1` leads `t2`, then
        `count` will peak in a positive time bin.
    """
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if auto: t2 = t1

    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)
    bins = np.linspace(-limit, limit, num=(2 * limit/bin_width + 1))

    # This is the old way to calculate bin edges. I think it is more
    # sensitive to numerical error. The new way may slightly change the
    # way that spikes near the bin edges are assigned.
    #bins = np.arange(-limit, limit + bin_width, bin_width)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to numpy.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] -= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins
