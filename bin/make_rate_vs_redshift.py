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

import ANTARES_object

import plasticc
import plasticc.get_data

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
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
    kwargs['columns']=['objid','ptrobs_min','ptrobs_max','hostgal_photoz']

    sntypes = plasticc.get_data.GetData.get_sntypes()
    getter = plasticc.get_data.GetData(data_release)

    fig2 = plt.figure(figsize=(15,10))
    ax1 = fig2.add_subplot(111)

    cmap = plt.cm.tab20
    nlines = len(sntypes.keys())
    color = iter(cmap(np.linspace(0,1,nlines)))

    redshift_range = np.arange(0, 3.01, 0.01)

    with PdfPages(f'{fig_dir}/rate_analysis/rate_vs_redshift_{data_release}_{out_field}.pdf') as pdf:
        for i,  model in enumerate(sntypes.keys()):
            
            kwargs['model'] = model 
            kwargs['big'] = True
            head = getter.get_lcs_headers(**kwargs)

            model_name = sntypes[model]
            
            head = list(head)
            nobs = len(head) 
            if nobs == 0:
                continue

            objid, _, _, redshift = zip(*list(head)) 

            objid = np.array(objid)
            redshift = np.array(redshift)
            c = next(color)
                
            try:
                density = gaussian_kde(redshift, bw_method='scott')
            except Exception as e:
                continue 

            fig1 = plt.figure(figsize=(15,10))
            ax2 = fig1.add_subplot(111)
            hist(redshift, bins='scott', normed=True, ax=ax2)
            ax2.plot(redshift_range, density(redshift_range), color=c)
            ax1.plot(redshift_range, density(redshift_range), color=c, label=model_name)
            ax2.set_xlabel('redshift', fontsize='xx-large')
            ax2.set_ylabel(model_name, fontsize='xx-large')
            ax2.set_xlim(0, 3.5)
            pdf.savefig(fig1)
        #end loop over models
        ax1.set_xlabel('redshift', fontsize='xx-large')
        ax1.set_ylabel('PDF', fontsize='xx-large')
        ax1.legend(frameon=False)
        ax1.set_xlim(0, 3.5)
        pdf.savefig(fig2)
    #close pdf fig



if __name__=='__main__':
    sys.exit(main())
