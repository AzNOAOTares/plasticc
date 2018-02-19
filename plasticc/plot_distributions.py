import os
import numpy as np
import matplotlib.pyplot as plt
from get_data import GetData
import multiprocessing as mp
import itertools

ROOT_DIR = os.getenv('PLASTICC_DIR')


def plot_light_curves(data_release, fig_dir=None, field='%', sntype='%', snid='%'):
    getdata = GetData(data_release)
    result = getdata.get_transient_data(field=field, sntype=sntype, snid=snid)
    n=0
    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = head
        print(n)
        n+=1
        for f in phot.columns:  # Filter names
            flt, mag, magerr, mjd = phot[f]

            fsave = plt.figure(f)
            plt.errorbar(mjd, mag, yerr=magerr, fmt='.')
            plt.legend(objid)
    try:
        for f in phot.columns:
            fsave.savefig("{0}/{1}_{2}_{3}_{4}.png".format(fig_dir, field, sntype, snid, f))
    except Exception as e:
        print("No results for {}_{}_{}.png".format(field, sntype, snid))


def get_class_distributions(field, sntype, getdata):
    """ Get population stats for each field and sntype """
    print(field, sntype)
    stats = {}

    # Get the number of objects for each sntype
    result = getdata.get_transient_data(field=field, sntype=sntype, get_num_lightcurves=True)
    stats['nobjects'] = (next(result), 0)
    print("GOT COUNTS", field, sntype)

    # Get other stats for each sntype
    n = 0
    mwebv_list, epoch_range_list = [], []
    cadence_list = {f: [] for f in ['i', 'r', 'Y', 'u', 'g', 'z']}
    result = getdata.get_transient_data(field=field, sntype=sntype, get_num_lightcurves=False)
    print("GOT RESULTS", field, sntype)
    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = head

        # Loop through the filter light curves in each spectrum
        for f in phot.columns:  # Filter names
            flt, mag, magerr, mjd = phot[f]
            mwebv_list.append(mwebv)
            epoch_range_list.append(mjd.max() - mjd.min())
            cadence_list[f].append(np.median(np.diff(mjd)))

            n += 1
            if n % 100 == 0:
                print(n)

    if n == 0:
        stats['mwebv'] = (0, 0)
        stats['epoch_range'] = (0, 0)
        stats['cadence'] = (0, 0)
    else:
        stats['mwebv'] = (np.mean(mwebv_list), np.std(mwebv_list))
        stats['epoch_range'] = (np.mean(epoch_range_list), np.std(epoch_range_list))
        cadence_list_all = list(itertools.chain.from_iterable(list(cadence_list.values())))  # Combine lists
        stats['cadence'] = (np.nanmean(cadence_list_all), np.nanstd(cadence_list_all))

    return stats, field, sntype


def get_distributions_multiprocessing(data_release, fig_dir):
    getdata = GetData(data_release)
    fields = ['DDF', 'WFD']
    sntypes, sntypes_map = getdata.get_sntypes()
    sntype_names = [sntypes_map[i] for i in sntypes]
    sntypes_and_fields = list(itertools.product(fields, sntypes))
    sntype_stats = {'nobjects': {'DDF': {}, 'WFD': {}}, 'mean_mwebv': {'DDF': {}, 'WFD': {}},
                    'mean_epoch_range': {'DDF': {}, 'WFD': {}}, 'mean_cadence': {'DDF': {}, 'WFD': {}}}
    # sntype_stats = {'nobjects': {'DDF': {}}, 'mwebv': {'DDF': {}},
    #                 'epoch_range': {'DDF': {}}, 'cadence': {'DDF': {}}}

    pool = mp.Pool()
    results = [pool.apply_async(get_class_distributions, args=(field, sntype, getdata)) for field, sntype in sntypes_and_fields]
    pool.close()
    pool.join()
    print("FINISHED POOLING")
    outputs = [p.get() for p in results]
    for out in outputs:
        stats, field, sntype = out
        for key, value in stats.items():
            sntype_stats[key][field][sntype] = value

    print("PLOTTING HISTOGRAMS")
    # Plot the histogram for each statistic
    for field in fields:
        for stat in sntype_stats:
            fig, ax = plt.subplots(figsize=(20, 10))
            y, yerr = list(zip(*sntype_stats[stat][field].values()))
            print(stat, y, yerr, len(sntypes))
            rects = ax.bar(range(len(sntypes)), y, yerr=yerr, align='center')
            ax.set_xticks(range(len(sntypes)))
            ax.set_xticklabels(sntype_names)
            ax.set_xlabel('sntype')
            ax.set_ylabel(stat)
            autolabel(rects, ax)
            fig.savefig("{0}/{1}_{2}.png".format(fig_dir, field, stat))
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

        ax.text(rect.get_x() + rect.get_width()/2., label_position, '%s' % float('%.3g' % height), ha='center', va='bottom')


if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    get_distributions_multiprocessing(data_release='20180112', fig_dir=fig_dir)
    print("PLOTTING LIGHTCURVES")
    plot_light_curves(data_release='20180112', fig_dir=fig_dir, field='DDF', sntype='4', snid='87287')
    print("PLOTTED LIGHTCURVES")

    plt.show()

