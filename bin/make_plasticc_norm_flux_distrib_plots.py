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
import plasticc.database
import make_index
import plasticc.get_data
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import scipy.stats
from scipy.stats import gaussian_kde, describe
import astropy.io.fits as afits
import pickle

def main():
    fig_dir = os.path.join(WORK_DIR, 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    colors = OrderedDict([('u','blueviolet'), ('g','green'), ('r','red'), ('i','orange'), ('z','black'), ('Y','gold')])

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    _ = kwargs.pop('model')
    field = kwargs.get('field')
    
    out_field = field
    if out_field == '%':
        out_field = 'all'

    sntypes = plasticc.get_data.GetData.get_sntypes()

    getter = plasticc.get_data.GetData(data_release)
    fields, dtypes  = getter.get_phot_fields()
    fields.append('SIM_MAGOBS')
    getter.set_phot_fields(fields, dtypes)

    fig1 = plt.figure(figsize=(15, 10))
    ax1 = fig1.add_subplot(1,1,1)

    fig2 = plt.figure(figsize=(15, 10))
    ax2 = fig2.add_subplot(1,1,1)

    fig3 = plt.figure(figsize=(15, 10))
    ax3 = fig3.add_subplot(1,1,1)

    #out_values = np.arange(-8, 8.1, 0.1)
    out_values = np.arange(-30, 30.1, 0.11)
    out_snr = np.arange(0, 200.1, 0.1)

    cmap = plt.cm.tab20
    nlines = len(sntypes.keys()) - 3 #there's three meta types
    color = iter(cmap(np.linspace(0,1,nlines)))

    with PdfPages(f'{fig_dir}/Flux_distrib_{data_release}_{out_field}.pdf') as pdf:
        for i,  model in enumerate(sntypes.keys()):


            kwargs['model'] = model
            lcdata = getter.get_lcs_data(**kwargs)
            if lcdata is None:
                continue

            sim_flux  = None 
            obs_flux  = None
            sig_flux = None 
            snr      = None

            nobj = 0
            for head, phot in lcdata:
                nobj += 1

                obsid, _, _ = head
                lc = getter.convert_pandas_lc_to_recarray_lc(phot)

                sim = 10**(0.4*(27.5 - lc['sim_magobs'])) 
                obs = lc['flux'] 
                flux_err = lc['dflux']

                if model >= 80:
                    tfield, tmodel, tbase, tid  = obsid.split('_')
                    header_dir = 'LSST_{}_MODEL{}'.format(tfield, tmodel)
                    header_file = 'LSST_{}_{}_HEAD.FITS.gz'.format(tfield, tbase)
                    header_file = os.path.join(DATA_DIR, data_release, header_dir, header_file)
                    header_data = make_index.get_file_data(header_file, extension=1)
                    snid = np.array([x.strip() for x in header_data['SNID']])
                    ind = (snid == tid)
                    tu = header_data['LCLIB(TEMPLATE_MAG_u)'][ind][0]
                    tg = header_data['LCLIB(TEMPLATE_MAG_g)'][ind][0]
                    tr = header_data['LCLIB(TEMPLATE_MAG_r)'][ind][0]
                    ti = header_data['LCLIB(TEMPLATE_MAG_i)'][ind][0]
                    tz = header_data['LCLIB(TEMPLATE_MAG_z)'][ind][0]
                    tY = header_data['LCLIB(TEMPLATE_MAG_Y)'][ind][0]
                    template_mag_lookup = {'u':tu, 'g':tg, 'r':tr, 'i':ti, 'z':tz, 'Y':tY}
                    temp_mag = np.array([template_mag_lookup.get(x) for x in lc['pb']])
                    temp_flux = 10**(0.4*(27.5 - temp_mag))
                else:
                    temp_flux = 0. 

                obs += temp_flux
                sn = np.abs(obs)/flux_err

                if obs_flux is None:
                    obs_flux = obs 
                    sim_flux = sim
                    sig_flux = flux_err 
                    snr = sn
                else:
                    obs_flux = np.concatenate((obs_flux, obs))
                    sim_flux = np.concatenate((sim_flux, sim))
                    sig_flux = np.concatenate((sig_flux, flux_err))
                    snr = np.concatenate((snr, sn))
            
            if nobj == 0:
                continue

            if obs_flux is None:
                continue 

            nobs = len(obs_flux)

            if nobs == 0:
                continue 
           
            norm_flux = (obs_flux - sim_flux)/sig_flux

            #if not (sntypes.get(model) in ('RRLyrae', 'Mdwarf', 'BSR', 'String')):
            #    continue 

            c = next(color)
            print(sntypes.get(model), nobj, describe(norm_flux))

            kernel = gaussian_kde(norm_flux)
            lpdf = kernel.pdf(out_values)
            lcdf = np.cumsum(lpdf)/np.sum(lpdf)

            k2 = gaussian_kde(snr)
            lsn = k2.pdf(out_snr)

            label = '{} ({:n}; {:n}) '.format(sntypes.get(model), nobj, nobs)

            ax1.plot(out_values, lpdf+i/10., color=c, label=label, lw=3)
            ax2.plot(out_values, lcdf+i/10, color=c, label=label, lw=3)

            if model < 80:
                fig4 = plt.figure(figsize=(15, 10))
                ax4 = fig4.add_subplot(1,1,1)
                ax4.plot(sim_flux, sig_flux, marker='o', color=c,  markersize=2, alpha=0.35, linestyle='None')
                ax4.set_xlabel('sim_fluxobs')
                ax4.set_ylabel('fluxcalerr')
                fig4.suptitle(label)
                fig4.tight_layout(rect=[0, 0.03, 1, 0.92])
                fig4.savefig('{}/{}_{}_{}_{}_noise_vs_flux.png'.format(fig_dir, sntypes.get(model), model, data_release, field))
                plt.close(fig4)
            ax3.plot(out_snr, lsn+i/10, color=c, label=label, lw=3)

        #ax1.set_xlim(-10, 10)
        #ax2.set_xlim(-10, 10)
        ax3.set_xscale('log')

        ax1.axvline(0., color='grey', linestyle='--', lw=2)
        ax1.set_xlabel(r'Normalized Flux Difference $\frac{F_{O} - F_{S}}{\sigma}$')
        ax1.set_ylabel('PDF')
        ax1.legend(frameon=False, fontsize='small')

        ax2.axvline(0., color='grey', linestyle='--', lw=2)
        ax2.set_xlabel(r'Normalized Flux Difference $\frac{F_{O} - F_{S}}{\sigma}$')
        ax2.set_ylabel('CDF')
        ax2.legend(frameon=False, fontsize='small')

        ax3.set_xlabel('SNR')
        ax3.set_ylabel('PDF')
        ax3.legend(frameon=False, fontsize='small')

        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)





            




if __name__=='__main__':
    sys.exit(main())
