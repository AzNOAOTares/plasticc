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
    if out_field == 'DDF':
        upper_lim = 0.101
        step = 0.001
    else:
        upper_lim = 0.81
        step = 0.01

    mwebv_range = np.arange(0, upper_lim, step)

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
        long_model_name = f'{model_name}_{model}'
    
        try:
            density = gaussian_kde(hz, bw_method='scott')
        except Exception as e:
            continue 
    
        ax1.plot(mwebv_range, density(mwebv_range), color=c, label=long_model_name)

    ax1.set_xlabel('MWEBV', fontsize='xx-large')
    ax1.set_ylabel('PDF', fontsize='xx-large')
    ax1.legend(frameon=False)
    ax1.set_xlim(0, upper_lim - step)
    fig1.tight_layout()
    fig1.savefig(f'{fig_dir}/rate_analysis/extinction_checks_{data_release}_{out_field}.pdf')
    plt.close(fig1)


if __name__=='__main__':
    sys.exit(main())
