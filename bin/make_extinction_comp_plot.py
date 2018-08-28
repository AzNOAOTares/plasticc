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
from scipy.stats import gaussian_kde, describe
from astropy.visualization import hist


def main():
    fig_dir = os.path.join(WORK_DIR, 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    _ = kwargs.pop('model')
    out_field = kwargs.get('field')
    kwargs['columns']=['objid','ptrobs_min','ptrobs_max','mwebv', 'mwebv_err', ]

    sntypes = plasticc.get_data.GetData.get_sntypes()
    getter = plasticc.get_data.GetData(data_release)


    cmap = plt.cm.tab20
    nlines = len(sntypes.keys())
    color = iter(cmap(np.linspace(0,1,nlines-3))) 
    fig1 = plt.figure(figsize=(15,10))
    ax1 = fig1.add_subplot(111)

    mwebv_range = np.arange(0, 1.01, 0.01)

    with PdfPages(f'{fig_dir}/rate_analysis/extinction_checks_{data_release}_{out_field}.pdf') as pdf:
        for i,  model in enumerate(sntypes.keys()):
            kwargs['model'] = model 
            kwargs['big'] = True
            head = getter.get_lcs_headers(**kwargs)

            model_name = sntypes[model]
            
            head = list(head)
            nobs = len(head) 
            if nobs <= 1:
                continue
            
            c = to_hex(next(color), keep_alpha=False)
            objid, _, _, hz, dhz = zip(*list(head)) 
            if nobs <= 2500:
                g2 = (sns.jointplot(hz, dhz, color=c, kind='scatter', xlim=(0, 1), height=8).set_axis_labels("mwebv", "mwebv_err"))
            else:
                g2 = (sns.jointplot(hz, dhz, color=c, kind='hex', xlim=(0,1), height=8).set_axis_labels("z", "mwebv_err"))
            fig2 = g2.fig
            long_model_name = f'{model_name}_{model}'
            fig2.suptitle(long_model_name)
            fig2.tight_layout(rect=[0, 0, 1, 0.97])

            try:
                density = gaussian_kde(hz, bw_method='scott')
            except Exception as e:
                continue 

            ax1.plot(mwebv_range, density(mwebv_range), color=c, label=long_model_name)

            pdf.savefig(fig2)
            plt.close(fig2)
        ax1.set_xlabel('MWEBV', fontsize='xx-large')
        ax1.set_ylabel('PDF', fontsize='xx-large')
        ax1.legend(frameon=False)
        ax1.set_xlim(0, 1)
        pdf.savefig(fig1)
        plt.close(fig1)


if __name__=='__main__':
    sys.exit(main())
