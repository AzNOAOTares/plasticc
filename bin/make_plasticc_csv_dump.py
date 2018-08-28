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
data_release = '20180827'


def task(entry):
    """
    Worker function to process a table of light curves from fits_files
    """
    this_file_lcs = {}
    cols = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'PHOTFLAG']
    # map the passbands to integers - include the extra space that's in Rick's files so we can skip stripping
    flt_map = {'u ':0,'g ':1,'r ':2, 'i ':3,'z ':4,'Y ':5}
    batch_key = entry[0]['object_id']

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

            lc = at.Table(phot_data)[cols]
            flt = np.array([flt_map[x] for x in lc['FLT']], dtype=np.uint8)
            mjd = lc['MJD']
            
            thislc = at.Table([mjd, flt, lc['FLUXCAL'], lc['FLUXCALERR'], lc['PHOTFLAG']],\
                    names=['mjd','passband','flux','flux_err','photflag'])
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
            this_obs_lines = ['{},{:.4f},{:d},{:.6f},{:.6f},{:d}'.format(obj, *x) for x in lc]  
            batch_lines += this_obs_lines
            nbatch += 1
    return batch_key, nbatch, batch_lines


def fixpath(filename, public=True):
    dirn = os.path.dirname(filename)
    basen = os.path.basename(filename)
    if public:
        dirn = os.path.join(dirn, 'public')
    else:
        dirn = os.path.join(dirn, 'private')
    if not os.path.exists(dirn):
        os.makedirs(dirn)

    fixn = os.path.join(dirn, basen)
    return fixn




def main():
    # how many files should we split the test output into
    nfiles = 10
    wfd_thresh = 1000000

    # get the options for the code and set the data_release globally (ugly) to
    # allow MultiPool to work
    kwargs = plasticc.get_data.parse_getdata_options()
    global data_release 
    data_release = kwargs.pop('data_release')
    getter = plasticc.get_data.GetData(data_release)

    # setup paths for output 
    dump_dir = os.path.join(WORK_DIR, 'csv_dump', data_release)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

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


    header_file = outfile.replace('.csv','_metadata.csv')
    header_file = fixpath(header_file)

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
    # current as of 20180827
    aggregate_types = {11:11, 2:2, 3:3, 12:2, 13:3, 14:2, 41:41, 43:43, 51:51,
            60:60, 61:99, 62:99, 63:99, 64:64, 70:70, 80:80, 81:81, 83:83,
            84:84, 90:99, 91:91, 92:99, 93:91}
    aggregate_names = {11:'SNIa-normal', 2:'SNCC-II', 3:'SNCC-Ibc', 12:'SNCC-II', 13:'SNCC-Ibc', 
                    14:'SNCC-II', 41:'SNIa-91bg', 43:'SNIa-x',51:'KN', 60:'SLSN-I', 61:'PISN', 
                    62:'ILOT', 63:'CART', 64:'TDE',70:'AGN', 80:'RRlyrae', 81:'Mdwarf', 83:'EBE', 
                    84:'MIRA', 90:'uLens-Binary', 91:'uLens-Point', 92:'uLens-STRING', 93:'uLens-Point'}

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

    # type 99 is not included in training
    train_types = set(aggregate_types.values())  - set([99,])
    train_types = list(train_types)

    # make sure that there are no "other" classes included in the training data
    if dummy == 'training':
        mask = np.array([True if x in train_types else False for x in out['sntype']])
        out = out[mask]

    # randomize the output type ID - keep rare as 99
    target_map_file = outfile.replace('.csv', '_targetmap.txt').replace('_test_set','').replace('_training_set','')
    target_map_file = fixpath(target_map_file, public=False)
    try:
        target_map_data = at.Table.read(target_map_file, format='ascii')
        train_types = target_map_data['train_types']
        target_types = target_map_data['target_types']
        print(f'Restoring Target Map from {target_map_file}')
    except Exception as e:
        target_types = np.random.choice(99, len(train_types), replace=False).tolist()
        target_map_data = at.Table([train_types, target_types], names=['train_types', 'target_types'])
        target_map_data.write(target_map_file, format='ascii.fixed_width', delimiter=' ', overwrite=True)
        print(f'Wrote target mapping to file {target_map_file}')

    target_map = dict(zip(train_types, target_types))
    target = np.array([target_map.get(x, 99) for x in out['sntype']])
    out['target'] = target
    print('Mapping as {}'.format(target_map))

    # orig map file is like target_map (and also private) but includes the rares
    orig_map_file = outfile.replace('.csv', '_origmap.txt').replace('_test_set','').replace('_training_set','')
    orig_map_file = fixpath(orig_map_file, public=False)
    if not os.path.exists(orig_map_file):
        orig = []
        aggregated = []
        mapped = []
        names = []
        for key, val in aggregate_types.items():
            name = aggregate_names.get(key, 'Rare')
            names.append(name)
            orig.append(key)
            aggregated.append(val)
            mapping = target_map.get(val, 99)
            mapped.append(mapping)
        orig_map_data = at.Table([orig, aggregated, mapped, names],\
                            names=['ORIG_NUM', 'MODEL_NUM', 'TARGET', 'MODEL_NAME'])
        orig_map_data.write(orig_map_file, format='ascii.fixed_width', delimiter=' ', overwrite=True)
        print(f'Wrote original mapping to file {orig_map_file}')
    
    # galactic objects have -9 as redshift - change to NaN
    # the numpy.isclose should have worked last time.... check this by hand.
    dummy_val = np.repeat(-9, len(out))
    ind = out['hostgal_photoz'] == -9.
    out['hostgal_photoz'][ind] = np.nan
    out['hostgal_photoz_err'][ind] = np.nan

    # figure out what fits files the data are in
    fits_files =[ "LSST_{0}_MODEL{1}/LSST_{0}_{2}_PHOT.FITS".format(*x.split('_')) for x in out['objid']] 
    fits_files = np.array(fits_files)

    # the object names have the model name in them, so we need to edit them
    # new name = <SNID>
    orig_name = out['objid']
    new_name = np.array([ x.split('_')[-1] for x in orig_name], dtype=np.int32)
    ddf_field = np.zeros(len(new_name))
    ind = new_name < wfd_thresh
    ddf_field[ind] = 1

    # preseve the mapping  between old name, new name and file name
    out['object_id'] = new_name
    out['filename'] = fits_files
    out['ddf_bool'] = ddf_field

    # sort things by object id - Rick has already randomized these, so we preserve his order.
    out.sort('object_id')
    del new_name 
    del fits_files
    del ddf_field
    del target 
    del new_type

    # if we are generating test data, save a truth table
    if dummy == 'training':
        pass
    else: 
        out_name = out['object_id']
        target   = out['target']

        # remove the model type from the output header that goes with the test data
        out.remove_column('target')

        truth_file = outfile.replace('_set.csv', '_truthtable.csv')
        truth_file = fixpath(truth_file, public=False)
        if os.path.exists(truth_file):
            os.remove(truth_file)

        truth_table = at.Table([out_name, target], names=['object_id','target'])
        truth_table.write(truth_file)
        print(f'Wrote {truth_file}')

    nmc = len(out)
    out_ind = np.arange(nmc)
    if dummy == 'training':
        batch_inds = np.array_split(out_ind, max(multiprocessing.cpu_count()-4, 0))
    else:
        # if this is test data, we want to break files up so that DDF and WFD
        # are in separate files and the number of files is split so we don't
        # have a giant CSV file
        batch_inds = []
        ind = np.where(out['object_id'] < wfd_thresh)[0]
        print('DDF objects {}'.format(len(ind)))
        batch_inds.append(ind)
        ind = np.where(out['object_id'] >= wfd_thresh)[0]
        print('WFD objects {}'.format(len(ind)))
        batch_inds += np.array_split(ind, nfiles)

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
    batch_ids = np.arange(len(batches)) + 1
    batch_keys = [x['object_id'][0] for x in batches]
    batch_map = dict(zip(batch_keys, batch_ids))

    # do the output
    if dummy == 'training':
        with MultiPool() as pool:
            with tqdm(total=nmc) as pbar:
                with open(outfile, 'w') as f:
                    f.write('object_id,mjd,passband,flux,flux_err,detected_bool\n')
                    # change to pool.imap so order is preserved in output file
                    for result in pool.imap(task, batches):
                        _, nbatch, batchlines = result
                        pbar.update(nbatch)
                        f.write('\n'.join(batchlines))
                        f.write('\n')
                gc.collect()
    else:
        with MultiPool() as pool:
            with tqdm(total=nmc) as pbar:
                for batch in batches:
                    batch_key = batch[0]['object_id']
                    batch_id = batch_map[batch_key]
                    batchfile = outfile.replace('.csv', f'_batch{batch_id}.csv')
                    batchfile = fixpath(batchfile)
                    ind = np.arange(len(batch))
                    mini_inds = np.array_split(ind, max(multiprocessing.cpu_count()-4, 0))
                    mini_batches = [batch[x] for x in mini_inds]
                    nbatch = 0
                    batchlines = []
                    for result in pool.imap(task, mini_batches):
                        _, mini_nbatch, mini_batchlines = result
                        nbatch += mini_nbatch
                        batchlines += mini_batchlines
                        pbar.update(mini_nbatch)
                    with open(batchfile, 'w') as f:
                        f.write('object_id,mjd,passband,flux,flux_err,detected_bool\n')
                        f.write('\n'.join(batchlines))
                        f.write('\n')
                    gc.collect()

    
    # write out the header
    out.remove_columns(['objid', 'ptrobs_min', 'ptrobs_max', 'filename', 'sntype'])
    out.rename_column('sim_dlmu','distance_modulus')
   
    cols = ['object_id','ra','decl', 'ddf_bool', 'mwebv', 'mwebv_err', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distance_modulus']

    if dummy == 'training':
        cols.append('target')
    out = out[cols]
    out.write(header_file, format='ascii.csv', overwrite=True)






if __name__=='__main__':
    sys.exit(main())
