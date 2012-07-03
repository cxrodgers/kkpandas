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