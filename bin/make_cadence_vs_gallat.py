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
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
from scipy.stats import gaussian_kde, describe
from astropy.coordinates import SkyCoord

def main():
    fig_dir = os.path.join(WORK_DIR, 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    colors = OrderedDict([('u','blueviolet'), ('g','green'), ('r','red'), ('i','orange'), ('z','black'), ('Y','gold')])
    hcolors = OrderedDict([('u','Purples'), ('g','Greens'), ('r','Reds'), ('i','Oranges'), ('z','Greys'), ('Y','Blues')])

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    _ = kwargs.pop('model')
    field = kwargs.get('field')
    kwargs['columns']=['objid','ptrobs_min','ptrobs_max','ra','decl']
    
    out_field = field
    if out_field == '%':
        out_field = 'all'

    sntypes = plasticc.get_data.GetData.get_sntypes()
    getter = plasticc.get_data.GetData(data_release)

    fig1 = plt.figure(figsize=(15,10))
    fig2 = plt.figure(figsize=(15,10))
    ax1 = fig1.add_subplot(1,1,1, projection='aitoff')

    time_array = np.arange(0., 101, 1.)
    current_y = {pb:np.zeros(len(time_array)) for pb in colors}

    cmap = plt.cm.tab20
    nlines = len(sntypes.keys())
    color = iter(cmap(np.linspace(0,1,nlines)))
    legend = []
    labels = []


    with PdfPages(f'{fig_dir}/cadence_analysis/cadence_vs_gallat_{data_release}_{out_field}.pdf') as pdf:
        for i,  model in enumerate(sntypes.keys()):
            model_name = sntypes[model]
            print(model_name)
            fig3 = plt.figure(figsize=(15,10))
        
            kwargs['model'] = model
            lcdata = getter.get_lcs_data(**kwargs)
            if lcdata is None:
                continue
        
            ra  = []
            dec = []

            cadence = {pb:[] for pb in colors}
        
            nobj = 0
            for head, phot in lcdata:
                nobj += 1
                if nobj % 1000 == 0:
                    print(model_name, nobj)
                obsid, _, _ , thisra, thisdec = head
                lc = getter.convert_pandas_lc_to_recarray_lc(phot)
                
                thisra  = float(thisra)
                thisdec = float(thisdec)

                c = SkyCoord(thisra, thisdec, unit='deg')
                gal = c.galactic
                l = gal.l.value 
                b = gal.b. value 

                ra.append(thisra)
                dec.append(thisdec)

                for pb in colors:
                    ind = lc['pb'] == pb 
                    if len(lc[ind]) > 1:
                        t = lc['mjd'][ind]
                        t = np.sort(t)
                        cad = np.median(np.diff(t))
                        cadence[pb].append((b, cad))
                    else:
                        continue 
                #end loop over pb
            #end loop over objects
            c = next(color)
            patch = mpatches.Patch(color=c, label=model_name)
            legend.append(patch)
            labels.append(model_name)

            ra = np.array(ra)
            dec = np.array(dec)
            nobs = len(ra) 
            if nobs == 0:
                continue
            if nobs > 10:
                alpha = 1./int(np.log10(nobs))
            if alpha <= 0.2:
                alpha/=10
            ax1.scatter(ra, dec, marker='.', color=c, alpha=alpha)
            for i, pb in enumerate(colors):
                pbcolor = colors[pb]
                thiscad = cadence[pb]
                
                thispbgallat, thispbcad = zip(*thiscad)
                thispbgallat = np.array(thispbgallat)
                thispbcad    = np.array(thispbcad)

                if len(thispbcad) < 3:
                    continue 

                thispbgallat = np.abs(thispbgallat)
                hcol = hcolors[pb]

                ax3 = fig3.add_subplot(3, 2, i+1)
                ntimebins = int(np.ceil((thispbcad.max() - thispbcad.min())/5.))
                if ntimebins < 5:
                    ntimebins = 5
                h = ax3.hist2d(thispbgallat, thispbcad, bins=[18,ntimebins], cmap=hcol, norm=LogNorm())
                fig3.colorbar(h[3], ax=ax3)
                ax3.set_xlabel('Gal Latitude')
                ax3.set_ylabel('{} Cadence'.format(pb))

                try:
                 kernel = gaussian_kde(thispbcad)
                except Exception as e:
                    continue
                ax2 = fig2.add_subplot(3, 2, i+1)
                this_y = current_y[pb]
                new_y = kernel(time_array)
                ax2.fill_between(time_array, new_y+this_y, this_y, color=c)
                current_y[pb] = this_y + new_y
            #end loop over pb
            fig3.suptitle('{} {}'.format(model_name, field), fontsize='large')
            fig3.tight_layout(rect=[0,0,1,0.93])
            #fig3.savefig('{}/cadence_analysis/{}_{}_cadence_vs_gallat.png'.format(fig_dir, model_name, field))
            pdf.savefig(fig3)
            print(nobj)
        #end loop over models
        for i, pb in enumerate(colors):
            ax2 = fig2.add_subplot(3, 2, i+1)
            ax2.set_xlabel('{} Cadence'.format(pb))
            ax2.set_ylabel('PDF')
        fig2.legend(legend, labels, ncol=5, fontsize='small', loc='upper center')
        fig2.tight_layout(rect=[0,0,1,0.93])
        pdf.savefig(fig2)
        fig1.legend(legend, labels, ncol=5, fontsize='small', loc='upper center')
        fig1.tight_layout(rect=[0,0,1,0.93])
        fig1.savefig('{}/cadence_analysis/sky_distribution_{}_{}.png'.format(fig_dir, field, data_release))
    #close pdf fig


if __name__=='__main__':
    sys.exit(main())
