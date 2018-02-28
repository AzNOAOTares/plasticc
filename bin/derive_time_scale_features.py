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
import argparse
import ANTARES_object
import plasticc
import plasticc.database
import plasticc.get_data
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict

def main():
    fig_dir = os.path.join(WORK_DIR, 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    colors = OrderedDict([('u','blueviolet'), ('g','green'), ('r','red'), ('i','orange'), ('z','black'), ('Y','gold')])

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')
    field = kwargs.get('field')
    model_id = kwargs.get('model')
    sntypes = plasticc.get_data.GetData.get_sntypes()
    model_name = sntypes.get(model_id)
    out_field = field
    if out_field == '%':
        out_field = 'all'

    getter = plasticc.get_data.GetData(data_release)
    fields, dtypes  = getter.get_phot_fields()
    fields.append('SIM_MAGOBS')
    getter.set_phot_fields(fields, dtypes)

    lcdata = getter.get_lcs_data(**kwargs)
    if lcdata is None:
        raise RuntimeError('could not get light curves')
    else:
        with PdfPages(f'{fig_dir}/{model_name}_{data_release}_{out_field}.pdf') as pdf:
            for head, phot in lcdata:
                objid, _, _ = head
                lc = getter.convert_pandas_lc_to_recarray_lc(phot)
                obsid = np.arange(len(lc))
                laobj = ANTARES_object.LAobject(objid, objid, lc['mjd'], lc['flux'], lc['dflux'],\
                                        obsid, lc['pb'], lc['zpt'], per=False, mag=False, clean=True)

                phase = laobj.get_phase(per=True)
                period = laobj.best_period
                # things we should save
                # (nobs_ugrizY, nobs) features in paper_ugrizY + period1-5 + power1-5
            




if __name__=='__main__':
    sys.exit(main())
