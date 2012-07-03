"""Methods to read KlustaKwik-formatted spike times into dataframe.

Low-level
---------
These are just simple (but efficient) wrappers around pandas reading
methods that add on column names, etc.

read_resfile
read_fetfile
read_clufile
load_spiketimes


Medium-level
------------
from_KK : auto finds all relevant files in directory, methods to choose
groups, memoization ...


High-level
----------
KK_Server : object for integrating across multiple sessions / directories
You train it on the locations of data and it deals with calling from_KK.
"""

import numpy as np
import pandas
import os.path
import glob
from KKFileSchema import KKFileSchema


# Bare-bones input/output functions for each filetype
# We use pandas loading functions because they are much faster for
# this type of data than equivalent functions in matplotlib
def read_resfile(filename):
    """Returns spiketimes in samples as Series"""
    return pandas.read_table(filename, names=['spike_time'])['spike_time']

def read_fetfile(filename, guess_time_column=True):
    """Reads features from fet file.
    
    If guess_time_column, will look at the last column and if it contains
    only positive values, then we assume it is the spike time.
    """
    with file(filename) as fi:
        n_features = int(fi.readline().strip())       
        table_sz = len(fi.readline().strip().split(' '))
    
    # Have to specify the names of the columns or it uses the first row
    # as the titles
    data = pandas.read_table(filename, sep=' ', skiprows=1, 
        names=['feature%d' % n for n in range(table_sz)])
    
    # Auto-guess whether the last column is a time (it probably is)
    if guess_time_column and np.all(data[data.columns[-1]] > 0):    
        data = data.rename(columns={data.columns[-1]: 'spike_time'}, copy=False)
    
    # Here is where code to drop unwanted features would go, based on
    # n_features
    
    return data

def read_clufile(filename):
    """Returns cluster ids as Series"""
    return pandas.read_table(filename, skiprows=1, names=['unit'])['unit']

def load_spiketimes(kfs_or_path, group, fs=None):
    """Given KKFileSchema or path to one, load spike times from group
    
    Returns Series
    """
    kfs = KKFileSchema.coerce(kfs_or_path)
    
    # check if res-files exist, which are faster to load
    if 'res' in kfs.available_filetypes:
        spiketimes = read_resfile(kfs.resfiles[group])
    elif 'fet' in kfs.available_filetypes:
        spiketimes = read_fetfile(kfs.fetfiles[group])['spike_time']
    else:
        raise ValueError("no available method to grab spike times")
    
    # optionally convert to seconds
    if fs:
        spiketimes = spiketimes / float(fs)    
    
    return spiketimes

# This is the main function to intelligently load data from KK files
def from_KK(basename='.', groups_to_get=None, group_multiplier=None, fs=None,
    verify_unique_clusters=True, add_group_as_column=True, 
    load_memoized=True, save_memoized=True):
    """Main function for loading KlustaKwik data.
    
    basename : path to, or basename of, files
    group : int or list of groups to get, otherwise get all groups
    group_multiplier : if None, the cluster ids are used as-is
        if int, then the group number times this multiplier is added to
        the cluster id.
        This is useful if groups contain the same cluster ids but you
        want them to have unique labels.
    fs : if None, the times are returned as integer number of samples
        otherwise, they are divided by this number
    verify_unique_clusters : if True, check that there are no overlapping
        cluster ids across groups
    add_group_as_column : if True, then the returned value has a column
        for the group from which the spike came.
    load_memoized : If a file like basename.kkp exists, load this DataFrame
        and return. Note all other parameters (except basename) are ignored.
    save_memoized : the data will be written to a file like
        basename.kkp after loading.
    
    Returns:
        DataFrame with columns 'unit', 'spike_time', and optionally 'group'
    """
    # load files like basename
    kfs = KKFileSchema.coerce(basename)
    
    # try to load memoized
    memoized_filename = kfs.basename + '.kkp'
    if load_memoized:
        try:
            data = pandas.load(memoized_filename)
            return_early = True
        except IOError:
            return_early = False
        
        if return_early:
            return data
    
    # which groups to get
    if groups_to_get:
        if not hasattr(groups_to_get, '__len__'):
            groups_to_get = [groups_to_get]
    else:
        groups_to_get = kfs.groups
    
    # get each group
    group_d = {}
    for group in groups_to_get:
        spiketimes = load_spiketimes(kfs, group, fs)
        
        if 'clu' in kfs.available_filetypes:
            unit_ids = read_clufile(kfs.clufiles[group])
        else:
            unit_ids = np.ones(spike_times.shape) * group
        
        if group_multiplier:
            unit_ids += group_multiplier * group
        
        # concatenate into data frame and add to dict
        if add_group_as_column:
            group_d[group] = pandas.DataFrame(
                {spiketimes.name: spiketimes, unit_ids.name: unit_ids,
                    'group': np.ones(len(spiketimes), dtype=np.int) * group})
        else:
            group_d[group] = pandas.DataFrame(
                {spiketimes.name: spiketimes, unit_ids.name: unit_ids})
    
    # optionally check if groups contain same cluster
    if verify_unique_clusters:
        clusters_by_group = [
            set(np.unique(np.asarray(groupdata.unit)))
            for groupdata in group_d.values()]
        n_unique_clusters = len(set.union(*clusters_by_group))
        n_total_clusters = sum([len(g) for g in clusters_by_group])
        if n_unique_clusters != n_total_clusters:
            raise ValueError("got %d overlapping clusters" % 
                (n_total_clusters - n_unique_clusters))
    
    # turn list into one giant dataframe for everybody
    sorted_keys = sorted(group_d.keys())
    data = pandas.concat([group_d[key] for key in sorted_keys], 
        ignore_index=True)    

    if save_memoized:
        data.save(memoized_filename)

    return data


class KK_Server:
    """Object to load spike data from multiple sessions (directories)"""
    def __init__(self, session_d=None, session_list=None, parent_dir=None, 
        group_multiplier=100, fs=30e3, **kk_kwargs):
        """Initialize a new server"""
        # Set up dict of sessions
        if session_d is None:
            session_d = {}
            for session in session_list:
                session_d[session] = os.path.join(parent_dir, session)
        self.session_d = session_d
        self.session_list = sorted(self.session_d.keys())
        
        # Set up calling kwargs
        self.kk_kwargs = kk_kwargs
        self.kk_kwargs['group_multiplier'] = group_multiplier
        self.kk_kwargs['fs'] = fs
    
    def load(self, session=None, group=None, unit=None, **kwargs):
        dirname = self.session_d[session]
        
        call_kwargs = self.kk_kwargs.copy()
        call_kwargs.update(kwargs)
        
        spikes = from_KK(dirname, load_memoized=True, save_memoized=True,
            **self.kk_kwargs)
        
        # Make this panda pick
        sub = spikes[spikes.unit == unit]
    
        return sub
    
    def save(self, filename):
        """Saves information for later use
        
        All that is necessary to reconsitute this object is session_d
        and kk_kwargs
        """
        import cPickle
        to_pickle = {
            'session_d': self.session_d, 
            'kk_kwargs': self.kk_kwargs}
        with file(filename, 'w') as fi:
            cPickle.dump(to_pickle, fi)
    
    def flush(self):
        """Delete all pickled data and start over"""
        pass
    
    @classmethod
    def from_saved(self, filename):
        """Load server from saved information"""
        import cPickle
        with file(filename) as fi:
            res = cPickle.load(fi)
        
        session_d = res['session_d']
        kk_kwargs = res['kk_kwargs']
        
        res = KK_Server(session_d=session_d)
        res.kk_kwargs = kk_kwargs
        return res