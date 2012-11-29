"""Pure Python object to handle filename parsing for KlustaKwik files"""

import numpy as np
import os.path
import glob

class KKFileSchema:
    def __init__(self, basename_or_dirname, ignore_tilde=True):
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
        
        If ignore_tilde, then any files ending in the '~' character are
        ignored, on the presumption that they are backups.
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
        self.ignore_tilde = ignore_tilde
        self.populate()
        
    def populate(self):
        """Find KK files in the directory and store
        
        Searches for files matching the following string:
            basename + '.' + extension + '.' + integer
        
        Stores all files that match this pattern in dicts keyed by extension.
        Checks that the group numbering of each extension matches up.
        """
        # find the files
        for ext in ['fet', 'clu', 'res', 'spk']:
            # Find everything with that sort of extension
            filename_l = sorted(glob.glob(self.basename + '.%s.*' % ext))
            
            # Optionally drop things ending in tilde
            if self.ignore_tilde:
                filename_l = filter(lambda s: not s.endswith('~'), filename_l)

            # Filter out anything that does not look like:
            #   basename + '.' + extension + '.' + int
            # with a warning
            filtered_filename_l, filenumber_l = [], []
            for fn in filename_l:
                pattern = '^%s\.%s\.(\d+)$' % (self.basename, ext)
                m = glob.re.match(pattern, fn)
                
                if m is None:
                    # Does not match, issue warning
                    print "warning: cannot parse %s, ignoring" % fn
                else:
                    # Add to filtered filename and filenumber lists
                    filenumber = int(m.groups()[0])
                    filtered_filename_l.append(fn)
                    filenumber_l.append(filenumber)
            
            # Store in _filenamed, dict from extension to file name list
            self._filenamed[ext] = filtered_filename_l
            
            # And _filenumberd, dict from extension to file numbers
            self._filenumberd[ext] = filenumber_l
        
        # check that files were actually found
        nonzero_exts = [
            key for key, val in self._filenamed.items() if len(val) > 0]
        if len(nonzero_exts) == 0:
            print "warning: no KK files found like %s" % self.basename
            self.groups = []
        else:
            # check that the group numbering of each extension matches up
            self.groups = self._filenumberd[nonzero_exts[0]]
            for ext in nonzero_exts:
                if self._filenumberd[ext] != self.groups:
                    print ("warning: KK extensions %s and %s out-of-sync" %
                        (ext, nonzero_exts[0]))
        
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