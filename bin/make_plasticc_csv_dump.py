#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from collections import OrderedDict
import sys
import time
import os
import numpy as np
import multiprocessing
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
import gc
import warnings
warnings.simplefilter('ignore')
from astropy.time import Time
data_release = '20180727'


def task(entry):
    """
    Worker function to process a table of light curves from fits_files
    """
    this_file_lcs = {}

    fits_files = entry['filename']
    uniq_files = np.unique(fits_files)
    for fits_file in uniq_files:
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

        ind = entry['filename'] == fits_file
        object_ptrs = entry[ind]
    
        for row in object_ptrs:
            objid = row['object_id']
            ptrobs_min = row['ptrobs_min']
            ptrobs_max = row['ptrobs_max']
            phot_data = data[ptrobs_min - 1:ptrobs_max]

            lc = at.Table(phot_data)
            flt = np.array([x.strip() for x in lc['FLT']])
            flt = flt.astype(bytes)
            mjd = lc['MJD']
            tai = Time(mjd, format='mjd', precision=7, scale='utc').isot
            
            thislc = at.Table([mjd, flt, lc['FLUXCAL'], lc['FLUXCALERR'], lc['PHOTFLAG'], tai],\
                    names=['mjd','passband','flux','flux_err','photflag', 'time_stamp'])
            ind = thislc['photflag'] != 1024
            if len(thislc[ind]) == 0:
                print(f'Fuck {objid} has no useful observations')
                continue
            thislc = thislc[ind]
            ind = thislc['photflag'] > 0
            thislc['photflag'][ind] = 1
            thislc.sort('mjd')
            this_file_lcs[objid] = thislc 
        phot_HDU.close()

    batch_lines = []
    nbatch = 0
    for obj in entry['object_id']:
        lc = this_file_lcs.get(obj, None)
        if lc is not None:
            this_obs_lines = ['{},{:.4f},{},{:.5E},{:.5E},{:d},{}'.format(obj, *x) for x in lc]  
            batch_lines += this_obs_lines
            nbatch += 1
    return nbatch, batch_lines


def main():
    # setup paths for output 
    dump_dir = os.path.join(WORK_DIR, 'csv_dump')
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
        outfile = os.path.join(dump_dir, 'plasticc_training_set.csv')
        offset = None
    else:
        if limit is None: 
            outfile = os.path.join(dump_dir, 'plasticc_test_set.csv')
        else:
            if offset is None:
                offset = 0
            outfile = os.path.join(dump_dir, 'plasticc_test_n{}_set.csv'.format(offset))
    header_file = outfile.replace('.csv','_header.csv')

    # make sure we remove any lingering files 
    if os.path.exists(outfile):
        os.remove(outfile)

    _ = kwargs.get('field')

    # set the header keywords for training and testing
    # same except for sntype will be removed from test and hostgal_photoz isn't
    # provided
    kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl', 'mwebv', 'mwebv_err',\
                        'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'sim_dlmu','sntype']

    # set an extrasql query to get just the DDF and WFD objects
    # sntype for testing = true sntype + 100 
    if dummy == 'training':
        extrasql = "AND sntype < 100 AND ((objid LIKE 'WFD%') OR (objid LIKE 'DDF%'))"
    else:
        extrasql = "AND sntype > 100 AND ((objid LIKE 'WFD%') OR (objid LIKE 'DDF%'))"

    # set up options for data retrieval ignoring many of the command-line
    # options - impose cuts later 
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
    # current as of 20180801
    aggregate_types = {1:1, 2:2, 3:3, 12:2, 13:3, 14:2, 41:41, 43:43, 51:51,
            60:60, 61:99, 62:99, 63:99, 64:64, 70:70, 80:80, 81:81, 83:83,
            84:84, 90:99, 91:91, 92:99, 93:91}
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

    # we're not necessariy keeping all the models we simulated - remove any models that are not in keep_types
    keep_types = aggregate_types.keys()
    mask = np.array([True if x in keep_types else False for x in out['sntype']])
    out = out[mask]

    # aggregate types - map to new class numbers (i.e. MODELNUM_PLASTICC)
    if dummy=='training':
        new_type = np.array([aggregate_types.get(x, None) for x in out['sntype']])
    else:
        new_type = np.array([aggregate_types.get(x, None) for x in out['sntype']])
    out['sntype'] = new_type 

    # make sure that there are no "other" classes included in the training data
    if dummy == 'training':
        # type 99 is not included in training
        train_types = set(aggregate_types.values())  - set([99,])
        train_types = list(train_types)
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

    # the object names have the model name in them, so we need to edit them
    # new name = <SNID>
    orig_name = out['objid']
    new_name = np.array([ x.split('_')[-1] for x in orig_name], dtype=np.int32)

    # preseve the mapping  between old name, new name and file name
    out['object_id'] = new_name
    out['filename'] = fits_files

    # sort things by object id - Rick has already randomized these, so we preserve his order.
    out.sort('object_id')

    # if we are generating test data, save a truth table
    if dummy == 'training':
        out.rename_column('sntype','class')
    else: 
        out_name = out['objid']
        sntype   = out['sntype']

        # remove the model type from the output header that goes with the test data
        out.remove_column('sntype')

        truth_file = outfile.replace('_set.csv', '_truthtable.csv')
        if os.path.exists(truth_file):
            os.remove(truth_file)

        truth_table = at.Table([out_name, sntype], names=['object_id','class'])
        truth_table.write(truth_file)
        print(f'Wrote {truth_file}')

    nmc = len(out)
    out_ind = np.arange(nmc)
    batch_inds = np.array_split(out_ind, multiprocessing.cpu_count())

    # make batches to load the data 
    batches = []
    for ind in batch_inds:
        this_batch_lcs = at.Table([out['object_id'][ind], 
                                    out['ptrobs_min'][ind], 
                                    out['ptrobs_max'][ind],
                                    out['filename'][ind]],
                                    names=['object_id','ptrobs_min', 'ptrobs_max', 'filename'])
        batches.append(this_batch_lcs)
    gc.collect()

    # do the output
    with MultiPool() as pool:
        with tqdm(total=nmc) as pbar:
            with open(outfile, 'w') as f:
                f.write('object_id,mjd,passband,flux,flux_err,detected_bool,time_stamp\n')
                for result in pool.imap_unordered(task, batches):
                    nbatch, batchlines = result
                    pbar.update(nbatch)
                    f.write('\n'.join(batchlines))
                    f.write('\n')
            gc.collect()

    
    # write out the header
    out.remove_columns(['objid', 'ptrobs_min', 'ptrobs_max', 'filename'])
    out.rename_column('sim_dlmu','distance_modulus')
    
    cols = ['object_id','ra','decl', 'mwebv', 'mwebv_err', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distance_modulus']

    if dummy == 'training':
        cols.append('class')
    out = out[cols]
    out.write(header_file, format='ascii.csv', overwrite=True)






if __name__=='__main__':
    sys.exit(main())
