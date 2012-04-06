"""Converter objects to read KlustaKwik-formatted spike times into dataframe.
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

def read_fetfile(filename):
    with file(filename) as fi:
        n_features = int(fi.readline().strip())       
        table_sz = len(fi.readline().strip().split(' '))
    
    # Have to specify the names of the columns or it uses the first row
    # as the titles
    data = pandas.read_table(filename, sep=' ', skiprows=1, 
        names=['feature%d' % n for n in range(table_sz)])
    
    # Auto-guess whether the last column is a time (it probably is)
    if np.all(data[data.columns[-1]] > 0):    
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
    verify_unique_clusters=True, concatenate_groups=True,
    add_group_as_column=True):
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
    concatenate_groups : if True, returns one DataFrame containing all
        information
        Otherwise, returns dict {groupnumber: groupDataFrame}
    add_group_as_column : if True, then the returned value has a column
        for the group from which the spike came.
    """
    # load files like basename
    kfs = KKFileSchema.coerce(basename)
    
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
    
    # optionally turn list into one giant dataframe for everybody
    if concatenate_groups:
        sorted_keys = sorted(group_d.keys())
        data = pandas.concat([group_d[key] for key in sorted_keys], 
            ignore_index=True)    
        return data
    else:
        return group_d
