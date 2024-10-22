#!/usr/bin/env python
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(WORK_DIR)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plasticc.get_data import GetData
from astropy.stats import sigma_clip
from collections import OrderedDict


def plot_light_curves(data_release, fig_dir=None, field_in='%', sntype_in='%', snid_in='%', limit=100, shuffle=False):
    getdata = GetData(data_release)
    result = getdata.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd'],\
            field=field_in, model=sntype_in, snid=snid_in, limit=limit, shuffle=shuffle, sort=False)

    t_plot, flux_plot, fluxerr_plot, peak_mjd_plot = {}, {}, {}, {}
    sntypes_map = getdata.get_sntypes()
    sntype_name = sntypes_map[sntype_in]

    peak_mjd_all = getdata.get_column_for_sntype(column_name='peakmjd', sntype=sntype_in, field=field_in)

    n = OrderedDict()
    n['u'] = 0
    n['g'] = 0
    n['r'] = 0
    n['i'] = 0
    n['z'] = 0
    n['Y'] = 0


    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd = head
        if all(n_value >= 6 for n_value in n.values()):
            break

        for f in n:  # Filter names
            data = phot.get(f)
            if data is None:
                continue
            flt, flux, fluxerr, mjd, photflag, zeropt = data
            ind = photflag != 1024
            flt = flt[ind]
            flux= flux[ind]
            fluxerr = fluxerr[ind]
            mjd = mjd[ind]
            photflag = photflag[ind]
            zeropt = zeropt[ind]
            t = mjd - peak_mjd


            filtered_err = sigma_clip(fluxerr, sigma=3., iters=5, copy=True)
            filtered_flux = sigma_clip(flux, sigma=7., iters=5, copy=True)
            bad1 = filtered_err.mask
            bad2 = filtered_flux.mask
            ind = ~np.logical_or(bad1, bad2)

            t = t[ind]
            flux = flux[ind]
            fluxerr = fluxerr[ind]

            if n[f] >= 1 and n[f] < 6:
                t_plot[f].append(t)
                flux_plot[f].append(flux)
                fluxerr_plot[f].append(fluxerr)
                peak_mjd_plot[f].append(peak_mjd)
                n[f] += 1
            elif n[f] == 0:
                t_plot[f] = [t,]
                flux_plot[f] = [flux,]
                fluxerr_plot[f] = [fluxerr,]
                peak_mjd_plot[f] = [peak_mjd,]
                n[f] += 1
            else:
                pass

    fig = plt.figure(figsize=(15, 10))
    for i, f in enumerate(n.keys()):
        ax = fig.add_subplot(3, 2, i + 1)
        ax.set_title(f)
        if f not in t_plot.keys():
            continue
        for j, _ in enumerate(t_plot[f]):
            ax.errorbar(t_plot[f][j], flux_plot[f][j], yerr=fluxerr_plot[f][j], marker='.', linestyle='None')
    fig.suptitle("{}".format(sntype_name))
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    return(fig)


if __name__ == '__main__':

    data_release = '20180715'
    field = 'WFD'
    limit=30
    shuffle=True

    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', data_release)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print("PLOTTING LIGHTCURVES")

    with PdfPages(f'{fig_dir}/all_{data_release}_{field}.pdf') as pdf:
        getdata = GetData(data_release)
        for s in getdata.get_sntypes():
            fig = plot_light_curves(data_release=data_release, fig_dir=fig_dir, field_in=field, sntype_in=s, snid_in='%', limit=limit, shuffle=shuffle)
            pdf.savefig(fig)
            plt.close(fig)
    print("PLOTTED LIGHTCURVES")
