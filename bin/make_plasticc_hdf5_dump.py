#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import time
import os
import numpy as np
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')
sys.path.append(WORK_DIR)
import plasticc
import plasticc.get_data
import astropy.table as at
from tqdm import tqdm 
import astropy.io.fits as afits 
from schwimmbad import MultiPool
import random
import h5py
import gc
data_release = '20180511'


def task(entry):
    """
    Worker function to process a single light curve from a file
    Accepts a tuple `entry` with object ID (GNDM format) and ptrobs_min,
    ptrobs_max for the original FITS file
    Loads the light curve, rectifies types and formatting inconsistencies and
    returns short object ID (without the model name) and light curve
    """
    fits_file, object_ptrs = entry
    phot_file = os.path.join(DATA_DIR, data_release, fits_file)
    if not os.path.exists(phot_file):
        phot_file = phot_file + '.gz'

    # open this file 
    try:
        phot_HDU = afits.open(phot_file, memmap=True)
    except Exception as e:
        message = f'Could not open photometry file {phot_file}'
        raise RuntimeError(message)
    data = phot_HDU[1].data

    # load all objects from this file
    this_file_lcs = {}
    for row in object_ptrs:
        objid = row['objid']
        ptrobs_min = row['ptrobs_min']
        ptrobs_max = row['ptrobs_max']
        phot_data = data[ptrobs_min - 1:ptrobs_max]
        lc = at.Table(phot_data)
        flt = np.array([x.strip() for x in lc['FLT']])
        flt = flt.astype(bytes)
        thislc = at.Table([lc['MJD'], flt, lc['FLUXCAL'], lc['FLUXCALERR'], lc['PHOTFLAG']],\
                names=['mjd','passband','flux','flux_err','photflag'])
        thislc.sort('mjd')
        this_file_lcs[objid] = thislc 
    phot_HDU.close()
    return this_file_lcs 


def main():
    # setup paths for output 
    dump_dir = os.path.join(WORK_DIR, 'hdf5_dump')
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # get the options for the code and set the data_release globally (ugly) to
    # allow MultiPool to work
    kwargs = plasticc.get_data.parse_getdata_options()
    global data_release 
    data_release = kwargs.pop('data_release')
    getter = plasticc.get_data.GetData(data_release)

    # we can use model as a dummy string to indicate if we are generating
    # training or test data
    dummy  = kwargs.pop('model')
    offset = kwargs.pop('offset')
    limit  = kwargs.pop('limit')

    if dummy == 'training':
        outfile = os.path.join(dump_dir, 'training_set.hdf5')
        offset = None
    else:
        if limit is None: 
            outfile = os.path.join(dump_dir, 'test_set.hdf5')
        else:
            if offset is None:
                offset = 0
            outfile = os.path.join(dump_dir, 'test_n{}_set.hdf5'.format(offset))

    # make sure we remove any lingering files 
    if os.path.exists(outfile):
        os.remove(outfile)

    _ = kwargs.get('field')

    # set the header keywords for training and testing
    # same except for sntype will be removed from test and hostgal_specz isn't
    # provided
    if dummy == 'training':
        kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl', 'mwebv', 'mwebv_err',\
                        'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'sntype']
    else:
        kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl', 'mwebv', 'mwebv_err',\
                        'hostgal_photoz', 'hostgal_photoz_err', 'sntype']

    # set an extrasql query to get just the DDF and WFD objects
    # sntype for testing = true sntype + 100 
    if dummy == 'training':
        extrasql = "AND sntype < 100 AND ((objid LIKE 'WFD%') OR (objid LIKE 'DDF%'))"
    else:
        extrasql = "AND sntype > 100 AND ((objid LIKE 'WFD%') OR (objid LIKE 'DDF%'))"

    # set up options for data retrieval ignoring many of the command-line
    # options
    kwargs['extrasql'] = extrasql
    kwargs['model'] = '%'
    kwargs['field'] = '%'
    kwargs['sort']  = True
    kwargs['shuffle'] = False
    kwargs['limit'] = None
    kwargs['get_num_lightcurves'] = True
    total = getter.get_lcs_headers(**kwargs)
    total = list(total)[0]
    kwargs['limit'] = total
    kwargs['get_num_lightcurves'] = False
    kwargs['offset'] = offset


    out = getter.get_lcs_headers(**kwargs)
    aggregate_types = {1:1, 2:2, 3:3, 12:2, 13:3, 14:2, 41:41, 43:43, 45:45, 51:51, 60:60, 61:61, 62:62, 63:63, 64:64, 80:80, 81:81, 82:82, 83:83, 84:84, 90:90, 91:91}
    if dummy == 'training':
        pass 
    else:
        aggregate_types = {x+100:y for x,y in aggregate_types.items()}
    print('Aggregating as ', aggregate_types)

    # make a big list of the header - NOTE THAT WE ALWAYS RETRIEVE ALL OBJECTS
    out = list(out)
    if dummy == 'training':
        # we don't need to shuffle the training set 
        pass
    else:
        # if we're generating test data, if we set a limit, just draw a random
        # sample else shuffle the full list
        if limit is not None:
            out = random.sample(out, limit)
        else:
            random.shuffle(out)

    # convert the selected header entries to a table
    out = at.Table(rows=out, names=kwargs['columns'])

    # we're not necessariy keeping all the models we simulated (42 and 50 are going bye bye)
    keep_types = aggregate_types.keys()
    mask = np.array([True if x in keep_types else False for x in out['sntype']])
    out = out[mask]

    # aggregate types
    if dummy=='training':
        new_type = np.array([aggregate_types.get(x, None) for x in out['sntype']])
    else:
        new_type = np.array([aggregate_types.get(x, None) for x in out['sntype']])
    out['sntype'] = new_type 

    # make sure that there are no "other" classes included in the training data
    if dummy == 'training':
        # not train types - 45, 60, 61, 62, 63, 64, 90, 91
        train_types = (1, 2, 3, 41, 43, 51, 80, 81, 82, 83, 84) 
        mask = np.array([True if x in train_types else False for x in out['sntype']])
        out = out[mask]

    # galactic objects have -9 as redshift - change to 0
    dummy_val = np.repeat(-9, len(out))
    ind = np.isclose(out['hostgal_photoz'], dummy_val)
    out['hostgal_photoz'][ind] = 0.
    out['hostgal_photoz_err'][ind] = 0.

    # figure out what fits files the data are in
    fits_files =[ "LSST_{0}_MODEL{1}/LSST_{0}_{2}_PHOT.FITS".format(*x.split('_')) for x in out['objid']] 
    fits_files = np.array(fits_files)
    uniq_files = np.unique(fits_files)

    # the object names have the model name in them, so we need to edit them
    # new name = <FIELD><SNID>
    orig_name = out['objid']
    new_name = [ '{}{}'.format(x.split('_')[0], x.split('_')[-1]) for x in orig_name]
    new_name = np.array(new_name)
    out_name = new_name
    nmc = len(out)

    # if we are generating test data, save a truth table
    if dummy == 'training':
        pass 
    else: 
        sntype = out['sntype']
        # remove the model type from the output header that goes with the test data
        out.remove_column('sntype')
        truth_file = outfile.replace('_set.hdf5', '_truthtable.hdf5')
        if os.path.exists(truth_file):
            os.remove(truth_file)
        # ... saving it in the truth table only
        orig_name = orig_name.astype(bytes)
        new_name  = new_name.astype(bytes)
        truth_table = at.Table([orig_name, new_name, sntype], names=['objid','shortid','sntype'])
        truth_table.write(truth_file, compression=True, path='truth_table', serialize_meta=False, append=True)

    # make batches to load the data 
    batches = {}
    for filename in uniq_files:
        ind = (fits_files == filename)
        this_fits_lcs = at.Table([out['objid'][ind], out['ptrobs_min'][ind], out['ptrobs_max'][ind]], names=['objid','ptrobs_min', 'ptrobs_max'])
        batches[filename] = this_fits_lcs 

    name_lookup = dict(zip(orig_name, out_name))

    gc.collect()

    # do the output
    failed = []
    with MultiPool() as pool:
        with tqdm(total=nmc) as pbar:
            for result in pool.imap_unordered(task, batches.items()):
                this_file_n = len(result.items())
                with h5py.File(outfile, 'a') as outf:
                   for true_obj, thislc in result.items():
                       short_obj = name_lookup[true_obj]
                       retries = 10
                       notwritten = True
                       overwrite = False
                       while notwritten and retries > 0:
                           try:
                               outf.create_dataset(short_obj, data=thislc, compression='lzf')
                               #thislc.write(outfile, path=short_obj, compression=True, serialize_meta=False, append=True,\
                               #    overwrite=overwrite)
                               notwritten = False
                           except Exception as e:
                               timer.sleep(0.010)
                               overwrite = True 
                               retries -= 1
                               print('{} {}'.format(true_obj, e))
                       if notwritten: 
                           failed.append((true_obj, short_obj))
                           print("Failed", true_obj)
                       outf.flush()
                   pbar.update(this_file_n)
            gc.collect()
    print(failed)
    
    # write out the header
    out['objid'] = out_name.astype(bytes) 
    out.remove_columns(['ptrobs_min', 'ptrobs_max'])
    out.write(outfile, compression=True, path='header', serialize_meta=False, append=True)






if __name__=='__main__':
    sys.exit(main())
