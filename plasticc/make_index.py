#!/usr/bin/env python
"""
Rick's files are in SNANA SIMLIB format and in FITS tables, with several
objects per photometry table, but the indices of each FITS table in a separate
header FITS file. 

This script makes a MySQL table/HDF5 file(?) for each data release with the
data for each object. We want to be able to: 

get_data(*object_IDs)
get_data(*field_IDs)
get_data(*model_names) 
get_data(*data_releases)
and have that return the right LCs as an iterator

"""

import sys
import os
import warnings
import glob
import astropy.io.fits as afits
import astropy.table as at
import numpy as np
import h5py
from . import database

ROOT_DIR = os.getenv('PLASTICC_DIR')
DIRNAMES = 1


def make_index_for_release(data_release, data_dir=None, redo=False):
    """
    Make an index for a single release
    """
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, 'plasticc_data')

    processed_table_file = os.path.join(data_dir, data_release, 'processed_{}.txt'.format(data_release))
    try:
        processed_tables = at.Table.read(processed_table_file, format='ascii.commented_header')
        used_files = processed_tables['filename'].tolist()
        if redo:
            raise RuntimeError('Clobbering')
    except Exception as e:
        used_files = []

    database.check_sql_table_for_release(data_release)

    filepattern = '*/*HEAD.FITS'
    fullpattern = os.path.join(data_dir, data_release, filepattern)
    files = glob.glob(fullpattern)

    header_fields = ['SNID', 'PTROBS_MIN', 'PTROBS_MAX', 'MWEBV', 'MWEBV_ERR','HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'SNTYPE', 'PEAKMJD']
    for header_file in files:
        # skip processed files
        if header_file in used_files:
            continue

        # sanity check for phot file existence
        phot_file = header_file.replace('HEAD', 'PHOT')
        if not os.path.exists(phot_file):
            message = 'Header file {} exists but no corresponding photometry file {}'.format(header_file, phot_file)
            warnings.warn(message, RuntimeWarning)
            continue
        dirname = os.path.split(os.path.dirname(header_file))[DIRNAMES]
        fieldname, modelname = dirname.replace('LSST_', '').split('_')
    
        # Save header data
        #header_HDU = afits.open(header_file)
        #header_data = header_HDU[1].data
        #header_out = [header_data[field] for field in header_fields]
        #header_out = [row.encode('UTF') if row.dtype == np.dtype('U16') else row for row in header_out]

        # we just need to go to MySQL here
        #header_out = at.Table(header_out, names=header_fields)
        print(header_file)

        used_files.append(header_file)

    out_table = at.Table()
    out_table['filename'] = used_files 
    out_table.write(processed_table_file, format='ascii.commented_header', overwrite=True)  
    return used_files 


def main():
    data_dir = os.path.join(ROOT_DIR, 'plasticc_data')
    for data_release in next(os.walk(data_dir))[DIRNAMES]:
        make_index_for_release(data_release, data_dir=data_dir, redo=True)



class GetData(object):
    """ Read .hdf5 file """
    def __init__(self, data_file, data_release):
        self.data = h5py.File(data_file, 'r')
        self.header = self.data[data_release]['header']
        self.phot = self.data[data_release]['phot']

    def get_field(self, fieldname):
        """Return a field from the .hdf5 file. E.g. get_field('SNID')"""
        try:
            return self.header[fieldname]
        except AttributeError:
            return self.phot[fieldname]























if __name__=='__main__':
    sys.exit(main())

