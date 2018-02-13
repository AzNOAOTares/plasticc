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


def main():
    saved_data_file = 'data.hdf5'
    # try:
    #     os.remove(saved_data_file)
    # except OSError:
    #     pass

    root_dir = os.getenv('PLASTICC_DIR')
    data_dir = os.path.join(root_dir, 'plasticc_data')
    DIRNAMES = 1
    filepattern = '*/*HEAD.FITS'

    header_fields = ['SNID', 'PTROBS_MIN', 'PTROBS_MAX', 'MWEBV', 'MWEBV_ERR','HOSTGAL_PHOTOZ', 'HOSTGAL_PHOTOZ_ERR', 'SNTYPE', 'PEAKMJD']
    phot_fields = ['MJD', 'FLT', 'MAG', 'MAGERR']

    for data_release in next(os.walk(data_dir))[DIRNAMES]:
        fullpattern = os.path.join(data_dir, data_release, filepattern)
        files = glob.glob(fullpattern)
        header_table = at.Table()
        phot_table = at.Table()
        for header_file in files:
            phot_file = header_file.replace('HEAD', 'PHOT')
            if not os.path.exists(phot_file):
                message = 'Header file {} exists but no corresponding photometry file {}'.format(header_file, phot_file)
                warnings.warn(message, RuntimeWarning)
                continue
            dirname = os.path.split(os.path.dirname(header_file))[DIRNAMES]
            fieldname, modelname = dirname.replace('LSST_', '').split('_')

            # Save header data
            header_HDU = afits.open(header_file)
            header_data = header_HDU[1].data
            header_out = [header_data[field] for field in header_fields]
            header_out = [row.encode('UTF') if row.dtype == np.dtype('U16') else row for row in header_out]  # Replace unicode types with utf strings before saving as HDF5
            header_out = at.Table(header_out, names=header_fields)
            header_table = at.vstack([header_table, header_out])  # Combine headers from each model

            # Save phot data
            photo_HDU = afits.open(phot_file)
            phot_data = photo_HDU[1].data
            phot_out = [phot_data[field] for field in phot_fields]
            phot_out = [row.encode('UTF') if row.dtype in (np.dtype('U16'), np.dtype('U2')) else row for row in phot_out]
            phot_out = at.Table(phot_out, names=phot_fields)
            phot_table = at.vstack([phot_table, phot_out])  # Combine headers from each model

        header_table.write(saved_data_file, path=data_release+'/header', compression=True)
        phot_table.write(saved_data_file, path=data_release+'/phot', append=True, compression=True)


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

