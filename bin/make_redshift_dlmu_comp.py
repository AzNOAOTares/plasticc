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
import astropy.table as at
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

def main():

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    fig_dir = os.path.join(WORK_DIR, 'Figures', data_release)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    _ = kwargs.pop('model')
    _ = kwargs.get('field')

    kwargs['model'] = '%'
    kwargs['field'] = '%'
    kwargs['columns']=['objid','hostgal_specz','hostgal_photoz', 'sim_redshift_host', 'sim_dlmu', 'redshift_helio', 'redshift_final', 'sim_redshift_cmb', 'sntype', 'snid']
    kwargs['extrasql'] = "AND sntype < 80 AND ((objid LIKE 'DDF%') or (objid LIKE 'WFD%'))"
    kwargs['big'] = True

    sntypes = plasticc.get_data.GetData.get_sntypes()
    getter = plasticc.get_data.GetData(data_release)
    head = getter.get_lcs_headers(**kwargs)
    
    head = list(head)
    nobs = len(head) 
    if nobs <= 1:
        message = 'Not enough observations to make plot. Check SQL.'
        raise RuntimeError(message)

    objid, sz, hz, tz, mu, solz, fz, cz,  target, snid = zip(*head)


    cosmo = FlatLambdaCDM(70, 0.3)
    snid = np.array(snid, dtype=np.int32)
    tz = np.array(tz)
    mu = np.array(mu)
    indz = tz.argsort()
    tmu = cosmo.distmod(tz).value
    deltamu = (tmu - mu) 
    outliers = (np.abs(deltamu) >= 0.3*(1+tz))

    out = at.Table([snid, tz, sz, hz, mu, tmu, deltamu],\
            names=['snid', 'sim_redshift_host', 'hostgal_specz', 'hostgal_photoz', 'sim_dlmu', 'calc_mu', 'deltamu']) 
    out = out[outliers]
    cols = out.dtype.names
    for col in cols[1:]:
        out[col].format= '%.4f'
    out.sort('snid')

    out.write(f'redshift_distmod_outliers_{data_release}.txt',\
            format='ascii.fixed_width', delimiter=' ', overwrite=True)


    fig_kw = {'figsize':(15, 10)}
    fig, ax = plt.subplots(2, 3, sharey=True, **fig_kw)
    ax[0][0].scatter(tz, mu, marker='.') 
    ax[0][0].plot(tz[indz], tmu[indz], linestyle='--', lw=0.5, color='C1', marker='.')
    ax[0][0].scatter(tz[outliers], mu[outliers], marker='.', color='C3') 
    ax[0][1].scatter(sz, mu, marker='.') 
    ax[0][2].scatter(hz, mu, marker='.') 
    ax[1][0].scatter(solz, mu, marker='.') 
    ax[1][1].scatter(cz, mu, marker='.') 
    ax[1][2].scatter(fz, mu, marker='.') 
    ax[0][0].set_xlabel('SIM_REDSHIFT_HOST')
    ax[0][1].set_xlabel('HOSTGAL_SPECZ')
    ax[0][2].set_xlabel('HOSTGAL_PHOTOZ')
    ax[1][0].set_xlabel('REDSHIFT_HELIO')
    ax[1][1].set_xlabel('SIM_REDSHIFT_CMB')
    ax[1][2].set_xlabel('REDSHIFT_FINAL')
    ax[0][0].set_ylabel('SIM_DLMU')
    ax[1][0].set_ylabel('SIM_DLMU')
    fig.tight_layout()
    out_file = os.path.join(fig_dir, f'redshift_mu_{data_release}_training_extragal.pdf')
    fig.savefig(out_file)
    plt.close(fig)

if __name__=='__main__':
    sys.exit(main())





