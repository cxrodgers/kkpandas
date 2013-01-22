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
import os
import glob
from KKFileSchema import KKFileSchema
import utility

SPIKE_TIME_COLUMN_NAME = 'time'

# Bare-bones input/output functions for each filetype
# We use pandas loading functions because they are much faster for
# this type of data than equivalent functions in matplotlib
def read_resfile(filename):
    """Returns spiketimes in samples as Series"""
    return pandas.read_table(
        filename, names=[SPIKE_TIME_COLUMN_NAME])[SPIKE_TIME_COLUMN_NAME]

def write_resfile(df, filename):
    """Returns spiketimes in samples as Series"""
    with file(filename, 'w') as fi:
        df.tofile(fi, sep="\n")
        fi.write('\n')


def read_fetfile(filename, guess_time_column=True, return_nfeatures=False):
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
        data = data.rename(columns={data.columns[-1]: SPIKE_TIME_COLUMN_NAME}, 
            copy=False)
    
    # Here is where code to drop unwanted features would go, based on
    # n_features
    
    if return_nfeatures:
        return data, n_features
    else:
        return data

def write_fetfile(df, filename, also_write_times=True, 
    count_time_as_feature=True):
    """Write out features to fetfile.
    
    also_write_times: Write spike times as another row of feature file
    count_time_as_feature: Include the spike time in the feature count

    Notes
    -----
    To view the files in Klusters, you must set both also_write_times
    and count_time_as_feature to True. This is a bug in Klusters though,
    bceause you wouldn't actually want to use the spike time as a feature
    for clustering.
    """
    if SPIKE_TIME_COLUMN_NAME not in df.columns and also_write_times:
        print "warning: no spike times provided to write"
        also_write_times = False
    
    cols = df.columns
    if also_write_times:
        if count_time_as_feature:
            n_features = df.shape[1]
        else:
            n_features = df.shape[1] - 1
    elif SPIKE_TIME_COLUMN_NAME in df.columns:
        # Drop the spike times before writing
        cols = cols.drop([SPIKE_TIME_COLUMN_NAME])
        n_features = df.shape[1] - 1
    else:
        n_features = df.shape[1]
    
    with file(filename, 'w') as fi:
        fi.write("%d\n" % n_features)
        df.to_csv(fi, sep=' ', cols=cols, header=False, index=False)

def read_clufile(filename):
    """Returns cluster ids as Series"""
    return pandas.read_table(filename, skiprows=1, names=['unit'])['unit']

def write_clufile(df, filename):
    """Write cluster DataFrame as a *.clu file"""
    nclusters = len(df.unique())
    
    with file(filename, 'w') as fi:
        fi.write("%d\n" % nclusters)
        df.tofile(fi, sep="\n")
        fi.write("\n")

def read_spkfile(filename, n_spikes=-1, n_samples=-1,  n_channels=-1):
    """Returns waveforms as 3d array (n_spk, n_samp, n_chan)
    You can leave at most one shape parameter as -1
    """
    waveforms = np.fromfile(filename, dtype=np.int16)
    return waveforms.reshape((n_spikes, n_samples, n_channels))

def write_spkfile(waveforms, filename):
    """Writes waveforms to binary file
    
    waveforms : 3d array (n_spk, n_samp, n_chan)
    
    It will be converted to int16 before writing.
    """
    waveforms.astype(np.int16).tofile(filename)


def load_spiketimes(kfs_or_path, group, fs=None):
    """Given KKFileSchema or path to one, load spike times from group
    
    Returns Series
    """
    kfs = KKFileSchema.coerce(kfs_or_path)
    
    # check if res-files exist, which are faster to load
    if 'res' in kfs.available_filetypes:
        spiketimes = read_resfile(kfs.resfiles[group])
    elif 'fet' in kfs.available_filetypes:
        spiketimes = read_fetfile(kfs.fetfiles[group])[SPIKE_TIME_COLUMN_NAME]
    else:
        raise ValueError("no available method to grab spike times")
    
    # optionally convert to seconds
    if fs:
        spiketimes = spiketimes / float(fs)    
    
    return spiketimes


def read_all_from_group(basename='.', group=1, n_samples=-1, n_spikes=-1,
    n_channels=-1):
    d = {}
    kfs = KKFileSchema.coerce(basename)
    res = read_resfile(kfs.resfiles[group])
    d['res'] = res
    clu = read_clufile(kfs.clufiles[group])
    d['clu'] = clu
    fet = read_fetfile(kfs.fetfiles[group])
    d['fet'] = fet
    
    if n_spikes == -1:
        n_spikes = len(res)
    spk = read_spkfile(kfs.spkfiles[group], n_spikes=n_spikes,
        n_channels=n_channels, n_samples=n_samples)
    d['spk'] = spk

    
    return d
    

# This is the main function to intelligently load data from KK files
def from_KK(basename='.', groups_to_get=None, group_multiplier=None, fs=None,
    verify_unique_clusters=True, add_group_as_column=True, 
    load_memoized=False, save_memoized=False,
    also_get_features=False, also_get_waveforms=False, n_samples=-1, n_channels=-1):
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
    also_get_features, also_get_waveforms : if True, then the returned
        value has columns for these as well.
    n_samples, n_channels : Only necessary if also_get_waveforms. Only
        one of these two parameters is necessary in that case.
    
    Memoization
    ---
    Loading is faster if it is done using the binary pandas save and load 
    functions than it is with the ASCII KlustaKwik format. For this reason
    you can specify that the data be saved as a pandas file, or loaded from
    a pandas file.
    
    These options now default to False because of the potential for accidental
    mis-use. The reason is that no checking is done whether the current
    parameters are the same as the previous ones, when the memoization was
    done.
    
    load_memoized : If a file like basename.kkp exists, load this DataFrame
        and return. Note all other parameters (except basename) are ignored.
    save_memoized : the data will be written to a file like
        basename.kkp after loading.
    
    Returns:
        DataFrame with columns 'unit', 'time', and optionally 'group'
    """
    memoized_filename = None # to be determined later, if necessary
    
    # load files like basename
    try:
        kfs = KKFileSchema.coerce(basename)
    except ValueError:
        # This occurs when no spike files are found, but there might still
        # be kkp files.
        load_memoized = True
        memoized_filename = glob.glob(os.path.join(basename, '*.kkp'))[0]
    
    # try to load memoized
    if load_memoized:
        if memoized_filename is None:
            memoized_filename = kfs.basename + '.kkp'        
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
        
        # optionally get features too
        if also_get_features:
            assert 'fet' in kfs.available_filetypes
            # Read the feature file
            fetfile = kfs.fetfiles[group]
            features = read_fetfile(
                fetfile, guess_time_column=True, return_nfeatures=False)
            
            # Pop off the time column since we don't need it
            features.pop('time')
            
            # Concatenate to df for this group
            assert len(features) == len(group_d[group])
            group_d[group] = pandas.concat([group_d[group], features], axis=1)
        
        # optionally get waveforms too
        if also_get_waveforms:
            assert 'spk' in kfs.available_filetypes
            # Read the spike file
            # We know the number of spikes, but we need either the number
            # of samples or the number of channels
            spkfile = kfs.spkfiles[group]
            waveforms = read_spkfile(spkfile, n_spikes=len(group_d[group]), 
                n_samples=n_samples, n_channels=n_channels)
            
            # Flatten, convert to dataframe, and concatenate to result
            nsamptot = waveforms.shape[1] * waveforms.shape[2]
            waveforms_df = pandas.DataFrame(
                waveforms.swapaxes(1, 2).reshape(waveforms.shape[0], nsamptot), 
                columns=['wf%d' % n for n in range(nsamptot)])
            group_d[group] = pandas.concat(
                [group_d[group], waveforms_df], axis=1)
    
    # optionally check if groups contain same cluster
    if verify_unique_clusters:
        clusters_by_group = [
            set(np.unique(np.asarray(groupdata.unit)))
            for groupdata in group_d.values()]
        if len(clusters_by_group) > 0:
            # find number of unique clusters
            # will error here if no clusters found
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

def flush(kfs_or_path, verbose=False):
    """Remove any memoized file (basename.kkp) from the directory."""    
    # Coerce to file schema
    kfs = KKFileSchema.coerce(kfs_or_path)    
    
    # Find the memoized file
    to_delete = kfs.basename + '.kkp'
    
    # Delete it if it exists
    if os.path.exists(to_delete):
        if verbose: print "deleting", to_delete
        os.remove(to_delete)
    else:
        if verbose: print "no memoized files to delete"
    

class KK_Server:
    """Object to load spike data from multiple sessions (directories)
    
    The from_KK class method works great for a single session or a small
    amount of data. Eventually you want to load from many different sessions
    easily. 
    
    The purpose of this object is to encapsulate the finding and I/O of
    KK files across sessions. Once it is initialized, you just specify the 
    session name and the unit that you want and it returns it. 
    
    You can also save it to disk and then load it later, without reinitializing
    all of the file locations.
    
    It also takes care of memoization, sampling rates, etc.
    """
    def __init__(self, session_d=None, session_list=None, parent_dir=None, 
        group_multiplier=100, fs=30e3, **kk_kwargs):
        """Initialize a new server from scratch
        
        session_d : dict {session_name: full_path_to_KK_dir}
        session_list : list of session names (keys to session_d)
        parent_dir : If session_d is None, looks for subdirectories like
            [os.path.join(parent_dir, session_name) 
                for session_name in session_list]
        group_multiplier, fs, **kk_kwargs : used in call to from_KK
        """
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
        self.kk_kwargs['load_memoized'] = True
        self.kk_kwargs['save_memoized'] = True
    
    def get(self, session=None, group=None, unit=None, **kwargs):
        """Returns spike times for specified session * unit
        
        Extra keywords override object defaults (eg group_multiplier, fs,
        memoization...)
        
        Current behavior is to always load and save memoized versions for
        best speed. This might change ...
        
        
        """        
        # Where the files are
        dirname = self.session_d[session]
        
        # Update the usual calling kwargs with any additional ones
        call_kwargs = self.kk_kwargs.copy()
        call_kwargs.update(kwargs)
        
        # Do the loading
        spikes = from_KK(dirname, **call_kwargs)
        
        # Make this panda pick
        #sub = spikes[spikes.unit == unit]
        sub = utility.panda_pick_data(spikes, group=group, unit=unit)
    
        return sub
    
    def load(self, filename):
        """Renamed to get to avoid confusion with "save" """
        raise DeprecationWarning("Use 'get' instead")
    
    def save(self, filename):
        """Saves information for later use
        
        All that is necessary to reconstitute this object is session_d
        and kk_kwargs
        """
        import cPickle
        to_pickle = {
            'session_d': self.session_d, 
            'kk_kwargs': self.kk_kwargs}
        with file(filename, 'w') as fi:
            cPickle.dump(to_pickle, fi)
    
    def flush(self, verbose=False):
        """Delete all memoized data in my session dict"""
        # Flush all sessions in the object
        for session, path in self.session_d.items():
            if verbose:
                print "flushing", session
            
            # Call the flush method from this module (not this object)
            flush(path, verbose)
    
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


# Utility function for testing something, also demonstrates the usage
# of the reading and writing methods.
def append_duplicated_spikes(data_dir, output_dir, groupnum, idxs, n_samples=24):
    """Appends a fake neuron of duplicated spikes.
    
    This is useful for testing whether some of the spikes are all in one
    part of the cluster, which might suggest drift or bad clustering.
    
    data_dir : klusters directory of original data (will not be modified)
    output_dir : klusters directory containing copy of original data
        (THIS ONE WILL BE MODIFIED!)
        Copy over all clu, fet, res, etc files to the new directory.
    
    groupnum : tetrode number, ie extension of klusters files to modify
    
    idxs : indexes of spikes to duplicate as a new cluster
        This functions doesn't know which unit you are trying to clone (if
        any), so the indexes should be indexes into ALL of the spikes from
        the group.
    
    It will extract the times, features, and waveforms of the indexed spikes,
    then append them to the end of the same files in output_dir.
    
    The new cluster has an ID one greater than previous max.
    """
    # find files
    kfs1 = KKFileSchema.coerce(data_dir)
    kfs2 = KKFileSchema.coerce(output_dir)
    
    # Duplicate clu
    clu = kkpandas.kkio.read_clufile(kfs1.clufiles[groupnum])
    newclunum = clu.max() + 1
    newclu = pandas.concat([clu, 
        pandas.Series(newclunum * np.ones(len(idxs)), dtype=np.int)], 
        ignore_index=True)
    kkpandas.kkio.write_clufile(newclu, kfs2.clufiles[groupnum])
    
    # Duplicate res
    res = kkpandas.kkio.read_resfile(kfs1.resfiles[groupnum])
    newres = pandas.concat([res, res.ix[idxs]], ignore_index=True)
    kkpandas.kkio.write_resfile(newres, kfs2.resfiles[groupnum])
    
    # Duplicate fet
    fet = kkpandas.kkio.read_fetfile(kfs1.fetfiles[groupnum])
    newfet = pandas.concat([fet, fet.ix[idxs]], ignore_index=True)
    kkpandas.kkio.write_fetfile(newfet, kfs2.fetfiles[groupnum])
    
    # Duplicate spk
    spk = kkpandas.kkio.read_spkfile(kfs1.spkfiles[groupnum], n_samples=24,
        n_spikes=fet.shape[0])
    newspk = np.concatenate([spk, spk[idxs, :]], axis=0)
    kkpandas.kkio.write_spkfile(newspk, kfs2.spkfiles[groupnum])
