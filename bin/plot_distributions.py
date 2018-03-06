#!/usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = os.getenv('PLASTICC_DIR')
MOD_DIR  = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(MOD_DIR)
from plasticc.get_data import GetData
import multiprocessing as mp
import itertools



def get_class_distributions(field, sntype, getdata):
    """ Get population stats for each field and sntype """
    print(field, sntype)
    stats = {}

    # Get the number of objects for each sntype
    result = getdata.get_lcs_headers(field=field, model=sntype, get_num_lightcurves=True)
    stats['nobjects'] = (result, 0)
    print("GOT COUNTS", field, sntype)

    # Get other stats for each sntype
    n = 0
    mwebv_list, epoch_range_list = [], []
    cadence_list = {f: [] for f in ['i', 'r', 'Y', 'u', 'g', 'z']}
    result = getdata.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'mwebv', 'sntype'], field=field, model=sntype, limit=1000)
    print("GOT RESULTS", field, sntype)
    bad_mags = []
    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, mwebv, sntype = head

        # Loop through the filter light curves in each spectrum
        for f in phot.columns:  # Filter names
            flt, flux, fluxerr, mjd, zeropt = phot[f]
            # remove fluxes with mag = 128
            g = np.where(flux > 0)[0]  # good indexes (where magnitude isn't 128)
            b = np.where(flux <= 0)[0]  # bad indexes (where magnitude isn't 128)
            if g.size == 0:
                bad_mags.append([objid, f, mjd[b], 'ALL'])
                continue
            elif len(g) != len(flux):
                bad_mags.append([objid, f, mjd[b]])

            flt, flux, fluxerr, mjd, zeropt = flt[g], flux[g], fluxerr[g], mjd[g], zeropt[g]
            
            mwebv_list.append(mwebv)
            epoch_range_list.append(np.max(mjd) - np.min(mjd))
            cadence_list[f].append(np.median(np.diff(mjd)))

            n += 1
            if n % 1000 == 0:
                print(n)

    if n == 0:
        stats['mean_mwebv'] = (0, 0)
        stats['mean_epoch_range'] = (0, 0)
        stats['mean_cadence'] = (0, 0)
    else:
        stats['mean_mwebv'] = (np.mean(mwebv_list), np.std(mwebv_list))
        stats['mean_epoch_range'] = (np.mean(epoch_range_list), np.std(epoch_range_list))
        cadence_list_all = list(itertools.chain.from_iterable(list(cadence_list.values())))  # Combine lists
        stats['mean_cadence'] = (np.nanmean(cadence_list_all), np.nanstd(cadence_list_all))

    return stats, field, sntype, bad_mags


def get_distributions_multiprocessing(data_release, fig_dir):
    getdata = GetData(data_release)
    fields = ['DDF', 'WFD']
    sntypes_map = getdata.get_sntypes()
    sntypes = sntypes_map.keys()
    sntype_names = [sntypes_map[i] for i in sntypes]
    sntypes_and_fields = list(itertools.product(fields, sntypes))
    sntype_stats = {'nobjects': {'DDF': {}, 'WFD': {}}, 'mean_mwebv': {'DDF': {}, 'WFD': {}},
                    'mean_epoch_range': {'DDF': {}, 'WFD': {}}, 'mean_cadence': {'DDF': {}, 'WFD': {}}}

    pool = mp.Pool()
    results = [pool.apply_async(get_class_distributions, args=(field, sntype, getdata)) for field, sntype in sntypes_and_fields]
    pool.close()
    pool.join()
    print("FINISHED POOLING")
    bad_mags = []
    outputs = [p.get() for p in results]
    for out in outputs:
        stats, field, sntype, bad_mags_part = out
        bad_mags += bad_mags_part
        for key, value in stats.items():
            sntype_stats[key][field][sntype] = value

    # Save Bad mags
    with open('bad_mags.txt', 'w') as f:
        for line in bad_mags:
            f.write("%s\n" % line)

    print("PLOTTING HISTOGRAMS")
    # Plot the histogram for each statistic
    for field in fields:
        for stat in sntype_stats:
            fig, ax = plt.subplots(figsize=(20, 10))
            y, yerr = list(zip(*sntype_stats[stat][field].values()))
            rects = ax.bar(range(len(sntypes)), y, yerr=yerr, align='center')
            ax.set_xticks(range(len(sntypes)))
            ax.set_xticklabels(sntype_names)
            ax.set_xlabel('sntype')
            ax.set_ylabel(stat)
            ax.set_ylim(bottom=0)
            autolabel(rects, ax)
            fig.savefig("{0}/distributions/{1}_{2}_{3}.pdf".format(fig_dir, field, stat, data_release))
    return sntype_stats


def autolabel(rects, ax):
    """ Labels on top of bar charts. """
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that; otherwise, put it inside the column.
        if p_height > 0.95:  # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position, '%s' % float('%.4g' % height), ha='center', va='bottom')


if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    get_distributions_multiprocessing(data_release='20180221', fig_dir=fig_dir)

    plt.show()






