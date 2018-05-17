#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(WORK_DIR)
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import astropy.table as at
import pymysql
from collections import OrderedDict
from plasticc.get_data import GetData
import ANTARES_object
import warnings
import snmachine.snfeatures as sf

DIRNAMES = 1

def make_light_curve_batches(data_release, field_in='%', model_in='%', batch_size=1000):
    """
    Get all the lightcurves for a data_release, field and model 
    """
    # explicit settings that should not be altered for batch processing since
    # we want to ensure each object is unique and processed exactly once

    SORT = True 
    NOSHUFFLE = False
    DEFAULT_BATCH_SIZE = 1000
    
    try:
        batch_size = int(batch_size)
        if batch_size <= 1:
            message = 'Batch size must be positive integer'
            raise ValueError(message)
    except (TypeError, ValueError) as e:
        message = '{}\nBatch size invalid - using default'.format(e)
        warnings.warn(message, RuntimeWarning)
        batch_size = DEFAULT_BATCH_SIZE

    getter = GetData(data_release)
    num_lightcurves = getter.get_lcs_headers(field=field_in, model=model_in, shuffle=NOSHUFFLE, sort=SORT,\
                                        get_num_lightcurves=True)
    num_batches = int(np.ceil(num_lightcurves/batch_size))
    offsets = np.arange(num_batches)*batch_size

    # serial process for test
    for offset in offsets:
        get_features_for_light_curve_batch(data_release, field_in=field_in, model_in=model_in, batch_size=batch_size, offset=offset)


    

def get_features_for_light_curve_batch(data_release, field_in='%', model_in='%', batch_size=1000,  offset=0):
    """
    Get BATCH_SIZE light curves beginning from OFFSET
    """
    SORT = True 
    NOSHUFFLE = False

    getter = GetData(data_release)
    result = getter.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd', 'sim_redshift_host', 'mwebv', 'sim_dlmu'],\
                                field=field_in, model=model_in,\
                                shuffle=NOSHUFFLE, sort=SORT,\
                                limit=batch_size, offset=offset)
    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd, redshift, mwebv, dlmu = head
        header = {'redshift':redshift, 'dlmu':dlmu, 'peakmjd':peak_mjd}

        lc = getter.convert_pandas_lc_to_recarray_lc(phot)
        obsid = np.arange(len(lc))

        laobj = ANTARES_object.LAobject(objid, objid, lc['mjd'], lc['flux'], lc['dflux'],\
                                    obsid, lc['photflag'], lc['pb'], lc['zpt'], header=header, ebv=mwebv, per=False, mag=False, clean=True)
        yield laobj




def main():
    get_features_for_light_curve_batch('20180511', field_in='WFD', model_in=1, batch_size=10)


if __name__=='__main__':
    sys.exit(main())
