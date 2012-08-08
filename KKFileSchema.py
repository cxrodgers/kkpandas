"""Pure Python object to handle filename parsing for KlustaKwik files"""

import numpy as np
import os.path
import glob

class KKFileSchema:
    def __init__(self, basename_or_dirname):
        """Initialize a new KK schema from directory or basename.
        
        If basename is not None, then there should exist files like
            basename.fet.*
            basename.clu.*
            basename.res.*
        
        If basename is None, then will try to auto guess the basename
        from the files in dirname. The first alphabetical fetfile in
        dirname will be used.
        
        A normalized, absolutized value for `basename` is stored, and
        its parent directory is stored in self.dirname.
        """
        # Decide whether input was a basename or a dirname
        if os.path.isdir(basename_or_dirname):
            # auto-get basename from dirname
            # sanitize dirname
            dirname = os.path.abspath(basename_or_dirname)
            
            # get basename from first fet file
            filenames = sorted(glob.glob(os.path.join(dirname, '*.fet.*')))            
            if len(filenames) == 0:
                filenames = sorted(glob.glob(os.path.join(dirname, '*.res.*')))            
                if len(filenames) == 0:
                    raise ValueError("cannot find KK files in %s" % dirname)
            
            basename = os.path.splitext(os.path.splitext(filenames[0])[0])[0]
        else:
            basename = os.path.abspath(basename_or_dirname)
        
        self.basename = basename
        self.dirname = os.path.split(self.basename)[0]
        self._force_reload = True
        self._filenamed = {}
        self._filenumberd = {}
        self.populate()
        
    def populate(self):
        # find the files
        for ext in ['fet', 'clu', 'res', 'spk']:
            self._filenamed[ext] = sorted(glob.glob(self.basename + 
                '.%s.*' % ext))
            
            filenumberstrings = [os.path.splitext(fn)[1] for fn in
                self._filenamed[ext]]
            
            try:
                self._filenumberd[ext] = map(lambda s: int(s[1:]),
                    filenumberstrings)
            except ValueError, TypeError:
                raise ValueError("cannot coerce group string to integer")
        
        # check existence
        nonzero_exts = [
            key for key, val in self._filenamed.items() if len(val) > 0]
        if len(nonzero_exts) == 0:
            print "warning: no KK files found like %s" % self.basename
            self.groups = []
        else:
            # check alignment
            self.groups = self._filenumberd[nonzero_exts[0]]
            for ext in nonzero_exts:
                if self._filenumberd[ext] != self.groups:
                    print ("warning: KK extensions %s and %s out-of-sync" %
                        ext, nonzero_exts[0])
        
        self._force_reload = False
    
    def filenames(self, ext='fet'):
        """Return filenames of a specified extension"""
        if self._force_reload:
            self.populate()
        return self._filenamed[ext]
    
    def filenumbers(self, ext='fet'):
        """Return filenumbers of a specified extension"""
        if self._force_reload:
            self.populate()
        return self._filenumberd[ext]
    
    @property
    def available_filetypes(self):
        if self._force_reload:
            self.populate()        
        return [
            key for key, val in self._filenamed.items() if len(val) > 0]
    
    @property
    def fetfiles(self):
        return dict(zip(self.filenumbers('fet'), self.filenames('fet')))
    
    @property
    def clufiles(self):
        return dict(zip(self.filenumbers('clu'), self.filenames('clu')))

    @property
    def resfiles(self):
        return dict(zip(self.filenumbers('res'), self.filenames('res')))

    @property
    def spkfiles(self):
        return dict(zip(self.filenumbers('spk'), self.filenames('spk')))

    @classmethod
    def coerce(self, kfs_or_path):
        """If you pass KKFileSchema, returns; otherwise initializes new"""
        # Coerce into KKFileSchema
        if hasattr(kfs_or_path, 'available_filetypes'):
            kfs = kfs_or_path
        else:
            kfs = KKFileSchema(kfs_or_path)
        return kfs