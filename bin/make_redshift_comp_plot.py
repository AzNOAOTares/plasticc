#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')
sys.path.append(WORK_DIR)
import numpy as np


import plasticc
import plasticc.get_data
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.backends.backend_pdf import PdfPages


def main():

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    fig_dir = os.path.join(WORK_DIR, 'Figures', data_release)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    _ = kwargs.pop('model')
    out_field = kwargs.get('field')
    kwargs['columns']=['objid','ptrobs_min','ptrobs_max','hostgal_photoz', 'hostgal_photoz_err', 'sim_redshift_host', ]

    sntypes = plasticc.get_data.GetData.get_sntypes()
    getter = plasticc.get_data.GetData(data_release)

    cmap = plt.cm.tab20
    keys = np.array(list(sntypes.keys()))
    nlines = len(keys[keys < 80])
    color = iter(cmap(np.linspace(0,1,nlines - 2))) 

    redshift_range = np.arange(0, 3.01, 0.01)

    with PdfPages(f'{fig_dir}/rate_analysis/redshift_checks_{data_release}_{out_field}.pdf') as pdf:
        for i,  model in enumerate(sntypes.keys()):
            if model >= 80:
                break
            
            kwargs['model'] = model 
            kwargs['big'] = True
            head = getter.get_lcs_headers(**kwargs)

            model_name = sntypes[model]
            
            head = list(head)
            nobs = len(head) 
            if nobs <= 1:
                continue
            
            c = to_hex(next(color), keep_alpha=False)
            objid, _, _, hz, dhz, z = zip(*head) 
            if nobs <= 2500:
                g1 = (sns.jointplot(z, hz, color=c, kind='scatter', xlim=(0, 3.), ylim=(0,3.), height=8).set_axis_labels("z", "hostz"))
                g2 = (sns.jointplot(hz, dhz, color=c, kind='scatter', xlim=(0, 3.), height=8).set_axis_labels("hostz", "hostz_err"))
            else:
                g1 = (sns.jointplot(z, hz, color=c, kind='hex', xlim=(0, 3.), ylim=(0,3.), height=8).set_axis_labels("z", "hostz"))
                g2 = (sns.jointplot(hz, dhz, color=c, kind='hex', xlim=(0,3.), height=8).set_axis_labels("z", "hostz_err"))
            fig1 = g1.fig 
            fig2 = g2.fig
            fig1.suptitle(f'{model_name}_{model}')
            fig2.suptitle(f'{model_name}_{model}')
            fig1.tight_layout(rect=[0, 0, 1, 0.97])
            fig2.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            plt.close(fig1)
            plt.close(fig2)


if __name__=='__main__':
    sys.exit(main())
