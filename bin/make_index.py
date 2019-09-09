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
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(WORK_DIR)
import warnings
import glob
import numpy as np
import astropy.io.fits as afits
import astropy.table as at
from plasticc import database

DIRNAMES = 1

def get_file_data(filename, extension=0):
    """
    Wrapped to handle getting the data that returns None if there there was an error
    """
    # get the header data
    try:
        data  = afits.getdata(filename, extension)
    except Exception as e:
        message = '{}\nSomething went wrong reading file {}. SKIPPING!'.format(e, filename)
        warnings.warn(message, RuntimeWarning)
        data = None
    return data


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
        processed_tables = at.Table.read(processed_table_file, names=('filename',),  format='ascii.no_header')
        used_files = processed_tables['filename'].tolist()
        if redo:
            raise RuntimeError('Clobbering')
        else:
            print('Restored {} used_files'.format(len(used_files)))
    except Exception as e:
        message = '{}\nSomething went wrong restoring processed file {}'.format(e, processed_table_file)
        warnings.warn(message, RuntimeWarning)
        used_files = []

    # if we have no list or we are clobbering, open the same file for output
    proc_flag = False
    if len(used_files) == 0:
        proc_table = open(processed_table_file, 'w')
        proc_flag = True

    # get a list of the header files in the data release
    filepattern = '*/*HEAD.FITS*'
    fullpattern = os.path.join(data_dir, data_release, filepattern)
    files = glob.glob(fullpattern)

    sample_file_for_schema = files[0]
    sample_data = get_file_data(sample_file_for_schema, extension=1)
    if sample_data is None:
        message = 'Something went wrong with with loading sample data. Cannot make schema'
        raise RuntimeError(message)

    header_fields, mysql_fields, mysql_schema = database.make_mysql_schema_from_astropy_bintable_cols(sample_data.columns)

    # make a mysql table for this data
    table_name = database.create_sql_index_table_for_release(data_release, mysql_schema, redo=redo)

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
        fieldname, modelname = dirname.replace('LSST_', '').replace('TEST_','').replace('IDEAL_','').replace('ZTF_','').replace(data_release+'_', '').split('_')
        modelname = modelname.replace('MODEL','')
        basename = basename.replace('LSST_','').replace('TEST_','').replace('IDEAL_','').replace('ZTF_','').replace(fieldname+'_','').replace('_HEAD.FITS.gz','').replace('_HEAD.FITS','')
    
        header_data = get_file_data(header_file, extension=1)
        if header_data is None:
            continue

        # fix the object IDs to be useful
        snid = header_data['SNID']
        newobjid = ['{}_{}_{}_{}'.format(fieldname, modelname, basename, x) for x in snid]
        newobjid = np.array(newobjid)
        header_out = [newobjid,]

        # get the rest of the useful fields from the header 
        try:
            header_out += [header_data[field] for field in header_fields]
            header_out = [row.encode('UTF') if row.dtype == np.dtype('U16') else row for row in header_out]
        except Exception as e:
            message = '{}\nSomething went wrong processing header file {}. SKIPPING!'.format(e, header_file)
            warnings.warn(message, RuntimeWarning)
            continue

        # fix the column names 
        header_out = at.Table(header_out, names=mysql_fields)
        header_out = np.array(header_out).tolist()
        
        # and save to mysql 
        try:
            nrows = database.write_rows_to_index_table(header_out, table_name)
        except Exception as e:
            message = '{}\nSomething went wrong writing header file {} to MySQL. SKIPPING!'.format(e, header_file)
            warnings.warn(message, RuntimeWarning)
            continue

        if nrows > 0:
            message = "Wrote {} from header file {} to MySQL table {}".format(nrows, header_file, table_name)
        else:
            message = "No useful data from header file {}".format(header_file)
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
        if data_release == 'src':
            continue 
        make_index_for_release(data_release, data_dir=data_dir, redo=False)


if __name__=='__main__':
    sys.exit(main())
