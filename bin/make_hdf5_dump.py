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
    dump_dir = os.path.join(WORK_DIR, 'hdf5_dump')
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    kwargs = plasticc.get_data.parse_getdata_options()
    global data_release 
    data_release = kwargs.pop('data_release')

    dummy = kwargs.pop('model')

    if dummy == 'training':
        outfile = os.path.join(dump_dir, 'training_set.hdf5')
    else:
        outfile = os.path.join(dump_dir, 'test_set.hdf5')

    if os.path.exists(outfile):
        os.remove(outfile)

    _ = kwargs.get('field')
    if dummy == 'training':
        kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl', 'mwebv', 'mwebv_err',\
                        'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'sntype']
    else:
        kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl', 'mwebv', 'mwebv_err',\
                        'hostgal_photoz', 'hostgal_photoz_err',]

    kwargs['model'] = '%'
    kwargs['field'] = '%'

    getter = plasticc.get_data.GetData(data_release)

    if dummy == 'training':
        extrasql = "AND sntype < 100 AND ((objid LIKE 'WFD%') OR (objid LIKE 'DDF%'))"
    else:
        extrasql = "AND sntype > 100 AND ((objid LIKE 'WFD%') OR (objid LIKE 'DDF%'))"

    kwargs['extrasql'] = extrasql
    head = getter.get_lcs_headers(**kwargs)
    head = list(head)
    if dummy == 'training':
        pass
    else:
        random.shuffle(head)

    out = at.Table(rows=head, names=kwargs['columns'])
    out.remove_columns(['ptrobs_min', 'ptrobs_max'])

    dummy_val = np.repeat(-9, len(out))
    ind = np.isclose(out['hostgal_photoz'], dummy_val)
    out['hostgal_photoz'][ind] = 0.
    out['hostgal_photoz_err'][ind] = 0.
    new_name = [ '{}{}'.format(x.split('_')[0], x.split('_')[-1]) for x in out['objid']]
    new_name = np.array(new_name)
    new_name = new_name.astype('bytes')
    out['objid'] = new_name 
    out.write(outfile, compression=True, path='header', serialize_meta=False, append=True)
    nmc = len(out)

    with MultiPool() as pool:
        with tqdm(total=nmc) as pbar:
            for result in pool.imap(task, head):
                short_obj, thislc = result 
                thislc.write(outfile, path=short_obj, compression=True, serialize_meta=False, append=True)
                pbar.update()
    






if __name__=='__main__':
    sys.exit(main())
