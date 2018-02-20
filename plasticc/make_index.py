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
import numpy as np
import astropy.io.fits as afits
import astropy.table as at
from . import database

ROOT_DIR = os.getenv('PLASTICC_DIR')
DIRNAMES = 1


def make_index_for_release(data_release, data_dir=None, redo=False):
    """
    Make an index for a single release
    """
    if data_dir is None:
        data_dir = os.path.join(ROOT_DIR, 'plasticc_data')

    used_files = []
    # get the list of files we processed already so we can skip in case something dies mid-processing
    processed_table_file = os.path.join(data_dir, data_release, 'processed_{}.txt'.format(data_release))
    try:
        processed_tables = at.Table.read(processed_table_file, names=('filename',),  format='no_header')
        used_files = processed_tables['filename'].tolist()
        if redo:
            raise RuntimeError('Clobbering')
    except Exception as e:
        used_files = []

    # if we have no list or we are clobbering, open the same file for output
    proc_flag = False
    if len(used_files) == 0:
        proc_table = open(processed_table_file, 'w')
        proc_flag = True

    # make a mysql table for this data
    table_name = database.create_sql_index_table_for_release(data_release, redo=redo)

    # get a list of the header files in the data release
    filepattern = '*/*HEAD.FITS*'
    fullpattern = os.path.join(data_dir, data_release, filepattern)
    files = glob.glob(fullpattern)

    # most of the header columns are junk - save only the relevant ones
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

        # do some string munging so we can set a more useful ID name
        basename = os.path.basename(header_file)
        dirname = os.path.split(os.path.dirname(header_file))[DIRNAMES]
        fieldname, modelname = dirname.replace('LSST_', '').split('_')
        modelname = modelname.replace('MODEL','')
        basename = basename.replace('LSST_','').replace(fieldname+'_','').replace('_HEAD.FITS.gz','').replace('_HEAD.FITS','')
    
        # get the header data
        try:
            header_HDU = afits.open(header_file)
            header_data = header_HDU[1].data
        except Exception as e:
            message = '{}\nSomething went wrong reading header file {}. SKIPPING!'.format(e, header_file)
            warnings.warn(message, RuntimeWarning)
            continue

        # fix the object IDs to be useful
        snid = header_data['SNID']
        newobjid = ['{}_{}_{}_{}'.format(fieldname, modelname, basename, x) for x in snid]
        newobjid = np.array(newobjid)
        header_out = [newobjid,]

        # get the rest of the useful fields from the header 
        try:
            header_out += [header_data[field] for field in header_fields[1:]]
            header_out = [row.encode('UTF') if row.dtype == np.dtype('U16') else row for row in header_out]
        except Exception as e:
            message = '{}\nSomething went wrong processing header file {}. SKIPPING!'.format(e, header_file)
            warnings.warn(message, RuntimeWarning)
            continue

        # fix the column names 
        keys = ['objid',] + [x.lower() for x in header_fields[1:]]
        header_out = at.Table(header_out, names=keys)
        header_out = np.array(header_out).tolist()
        
        # and save to mysql 
        try:
            nrows = database.write_rows_to_index_table(header_out, table_name)
        except Exception as e:
            message = '{}\nSomething went wrong writing header file {} to MySQL. SKIPPING!'.format(e, header_file)
            warnings.warn(message, RuntimeWarning)
            continue

        message = "Wrote {} from header file {} to MySQL table {}".format(nrows, header_file, table_name)
        print(message)

        if proc_flag:
            # make an entry in the processed file table for this file
            proc_table.write(header_file+'\n')

        # save that we've already processed this file
        used_files.append(header_file)
    
    if proc_flag:
        proc_table.close()
    return used_files 


def main():
    data_dir = os.path.join(ROOT_DIR, 'plasticc_data')
    for data_release in next(os.walk(data_dir))[DIRNAMES]:
        make_index_for_release(data_release, data_dir=data_dir, redo=False)
