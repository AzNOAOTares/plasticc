#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
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
data_release = '20180511'

def get_light_curve_array(objid, ptrobs_min, ptrobs_max):
    """ Get lightcurve from fits file as an array - avoid some Pandas overhead

    Parameters
    ----------
    objid : str
        The object ID. E.g. objid='DDF_04_NONIa-0004_87287'
    ptrobs_min : int
        Min index of object in _PHOT.FITS.
    ptrobs_max : int
        Max index of object in _PHOT.FITS.

    Return
    -------
    phot_out: pandas DataFrame
        A DataFrame containing the MJD, FLT, FLUXCAL, FLUXCALERR, ZEROPT seperated by each filter.
        E.g. Access the magnitude in the z filter with phot_out['z']['MAG'].
    """
    field, model, base, snid = objid.split('_')
    if field == 'IDEAL':
        filename = "{0}_MODEL{1}/{0}_{2}_PHOT.FITS".format(field, model, base)
    else:
        filename = "LSST_{0}_MODEL{1}/LSST_{0}_{2}_PHOT.FITS".format(field, model, base)
    phot_file = os.path.join(DATA_DIR, data_release, filename)
    if not os.path.exists(phot_file):
        phot_file = phot_file + '.gz'

    try:
        phot_HDU = afits.open(phot_file, memmap=True)
    except Exception as e:
        message = f'Could not open photometry file {phot_file}'
        raise RuntimeError(message)

    phot_data = phot_HDU[1].data[ptrobs_min - 1:ptrobs_max]
    phot_data = at.Table(phot_data)
    return phot_data

def task(entry):
    """
    Worker function to process a single light curve from a file
    Accepts a tuple `entry` with object ID (GNDM format) and ptrobs_min,
    ptrobs_max for the original FITS file
    Loads the light curve, rectifies types and formatting inconsistencies and
    returns short object ID (without the model name) and light curve
    """
    objid, ptrobs_min, ptrobs_max = entry[0:3]
    lc = get_light_curve_array(objid, ptrobs_min, ptrobs_max)
    objid_comp = objid.split('_')
    short_obj = '{}{}'.format(objid_comp[0], objid_comp[-1])
    
    flt = np.array([x.strip() for x in lc['FLT']])
    flt = flt.astype(bytes)
    thislc = at.Table([lc['MJD'], flt, lc['FLUXCAL'], lc['FLUXCALERR'], lc['PHOTFLAG']],\
                names=['mjd','passband','flux','flux_err','photflag'])
    thislc.sort('mjd')
    return (short_obj, thislc)



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


    head = getter.get_lcs_headers(**kwargs)

    # make a big list of the header - NOTE THAT WE ALWAYS RETRIEVE ALL OBJECTS
    head = list(head)
    if dummy == 'training':
        pass
    else:
        # if we're generating test data, if we set a limit, just draw a random
        # sample else shuffle the full list
        if limit is not None:
            head = random.sample(head, limit)
        else:
            random.shuffle(head)

    # convert the selected header entries to a table and remove uncessary columns    
    out = at.Table(rows=head, names=kwargs['columns'])
    out.remove_columns(['ptrobs_min', 'ptrobs_max'])

    # galactic objects have -9 as redshift - change to 0
    dummy_val = np.repeat(-9, len(out))
    ind = np.isclose(out['hostgal_photoz'], dummy_val)
    out['hostgal_photoz'][ind] = 0.
    out['hostgal_photoz_err'][ind] = 0.

    # the object names have the model name in them, so we need to edit them
    # new name = <FIELD><SNID>
    orig_name = out['objid']
    new_name = [ '{}{}'.format(x.split('_')[0], x.split('_')[-1]) for x in orig_name]
    new_name = np.array(new_name)
    new_name = new_name.astype('bytes')
    out['objid'] = new_name 

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
        sntype    = sntype.astype(bytes)
        truth_table = at.Table([orig_name, new_name, sntype], names=['objid','shortid','sntype'])
        truth_table.write(truth_file, compression=True, path='truth_table', serialize_meta=False, append=True)

    # write out the header
    out.write(outfile, compression=True, path='header', serialize_meta=False, append=True)
    nmc = len(out)

    # use a multiprocessing pool to load each light curve and dump to HDF5
    with MultiPool() as pool:
        with tqdm(total=nmc) as pbar:
            for result in pool.imap(task, head):
                short_obj, thislc = result 
                thislc.write(outfile, path=short_obj, compression=True, serialize_meta=False, append=True)
                pbar.update()
    






if __name__=='__main__':
    sys.exit(main())
