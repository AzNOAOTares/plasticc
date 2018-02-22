#!/usr/bin/env python
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(WORK_DIR)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plasticc
from plasticc.get_data import GetData



def plot_light_curves(data_release, fig_dir=None, field_in='%', sntype_in='%', snid_in='%', cadences=None):
    getdata = GetData(data_release)
    result = getdata.get_transient_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd'], field=field_in, sntype=sntype_in, snid=snid_in)

    t_plot, flux_plot, fluxerr_plot, peak_mjd_plot = {}, {}, {}, {}
    sntypes_map = getdata.get_sntypes()
    sntype_name = sntypes_map[sntype_in]
    med_cad = cadences[sntype_name]

    peak_mjd_all = getdata.get_column_for_sntype(column_name='peakmjd', sntype=sntype_in, field=field_in)
    non_transients = ['RRLyrae', 'Mdwarf', 'Mira']

    n = {'Y': 0, 'g': 0, 'i': 0, 'r': 0, 'u': 0, 'z': 0}

    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd = head
        if all(n_value >= 6 for n_value in n.values()):
            break

        for f in phot.columns:  # Filter names
            flt, flux, fluxerr, mjd, zeropt = phot[f]
            t = mjd - peak_mjd

            sn = np.abs(flux/fluxerr)
            ind = np.where(sn >= 1.)[0]

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
    for i, f in enumerate(t_plot.keys()):
        ax = fig.add_subplot(3, 2, i + 1)
        fig.tight_layout()
        fig.suptitle("{}".format(sntype_name))
        ax.set_title(f)
        if f not in t_plot.keys():
            continue
        for j, _ in enumerate(t_plot[f]):
            ax.errorbar(t_plot[f][j], flux_plot[f][j], yerr=fluxerr_plot[f][j], marker='.', linestyle='None')
    fig.savefig("{0}/{1}_{2}.pdf".format(fig_dir, field_in, sntype_name))
    return(fig)


if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    data_release = '20180221'

    print("PLOTTING LIGHTCURVES")

    epoch_ranges = {'SN1a': 112.5, 'CC': 112.1, 'SNIbc': 110.9, 'IIn': 115.6, 'SNIa-91bg': 111.0, 'pointIa': 106.1,
                    'Kilonova': 0., 'Magnetar': 560.9, 'PISN': 252.5, 'ILOT': 116.6, 'CART': 488.6,
                    'RRLyrae': 818.4, 'Mdwarf': 804.5}

    median_cadences = {'SN1a': 7.279, 'CC': 7.525, 'SNIbc': 7.621, 'IIn': 7.102, 'SNIa-91bg': 7.855, 'pointIa': 8.835,
                    'Kilonova': 0., 'Magnetar': 7.145, 'PISN': 7.256, 'ILOT': 6.987, 'CART': 8.502,
                    'RRLyrae': 18.37, 'Mdwarf': 21.79, 'Mira': 7., 'BSR': 7., 'String': 7}


    with PdfPages(f'{fig_dir}/all_{data_release}.pdf') as pdf:
        for s in [1, 2, 3, 4, 42, 45, 50, 60, 61, 62, 63, 80, 81, 82]:
            fig = plot_light_curves(data_release=data_release, fig_dir=fig_dir, field_in='DDF', sntype_in=s, snid_in='%', cadences=median_cadences)
            pdf.savefig(fig)
            plt.close(fig)
    print("PLOTTED LIGHTCURVES")
