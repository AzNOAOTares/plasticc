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
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(1,1,1)
                for i, pb in enumerate(colors):
                    m = laobj.passband == pb
                    ax.errorbar(phase[m], laobj.flux[m], yerr=laobj.fluxErr[m], marker='o', color=colors.get(pb, 'black'), linestyle='None', label=pb)
                    ax.set_ylabel('Flux')
                    ax.set_xlabel('phase', fontsize='large')
                ax.legend(frameon=False, fontsize='small')
                fig.suptitle('{} ({:.7f} days)'.format(objid, period), fontsize='x-large')
                fig.tight_layout(rect=[0, 0.03, 1, 0.9])
                pdf.savefig(fig)
                plt.close(fig)



if __name__=='__main__':
    sys.exit(main())
