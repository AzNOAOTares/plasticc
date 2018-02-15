import os
import numpy as np
import matplotlib.pyplot as plt
from get_data import GetData
import multiprocessing as mp
import itertools

ROOT_DIR = os.getenv('PLASTICC_DIR')


def get_class_distributions(field, sntype, getdata):
    """ Get population stats for each field and sntype """
    print(field, sntype)
    stats = {}

    # Get the number of objects for each sntype
    result = getdata.get_transient_data(field=field, sntype=sntype, get_num_lightcurves=True)
    stats['nobjects'] = next(result)

    # Get other stats for each sntype
    n, sum_mwebv, sum_epoch_range, sum_cadence = 0, 0, 0, 0
    result = getdata.get_transient_data(field=field, sntype=sntype, get_num_lightcurves=False)
    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = head
        mjd, flt, mag, magerr = phot

        sum_mwebv += mwebv
        sum_epoch_range += (mjd.max() - mjd.min())
        sum_cadence += np.median(np.diff(mjd))

        n += 1
        if n % 100 == 0:
            print(n, sum_mwebv, sum_epoch_range, sum_cadence)

    if n == 0:
        stats['mean_mwebv'] = 0
        stats['mean_epoch_range'] = 0
        stats['mean_cadence'] = 0
    else:
        stats['mean_mwebv'] = sum_mwebv / n
        stats['mean_epoch_range'] = sum_epoch_range / n
        stats['mean_cadence'] = sum_cadence / n

    return stats, field, sntype


def get_distributions_multiprocessing(data_release, fig_dir):
    getdata = GetData(data_release)
    fields = ['DDF', 'WFD']
    sntypes = getdata.get_sntypes()
    sntypes_and_fields = list(itertools.product(fields, sntypes))
    sntype_stats = {'nobjects': {'DDF': {}, 'WFD': {}}, 'mean_mwebv': {'DDF': {}, 'WFD': {}},
                    'mean_epoch_range': {'DDF': {}, 'WFD': {}}, 'mean_cadence': {'DDF': {}, 'WFD': {}}}

    pool = mp.Pool()
    results = [pool.apply_async(get_class_distributions, args=(field, sntype, getdata)) for field, sntype in sntypes_and_fields]
    pool.close()
    pool.join()

    outputs = [p.get() for p in results]
    for out in outputs:
        stats, field, sntype = out
        for key, value in stats.items():
            sntype_stats[key][field][sntype] = value

    # Plot the histogram for each statistic
    for field in fields:
        for stat in sntype_stats:
            plt.figure()
            rects = plt.bar(range(len(sntypes)), sntype_stats[stat][field].values(), align='center')
            plt.xticks(range(len(sntypes)), sntypes)
            plt.xlabel('sntype')
            plt.ylabel(stat)
            autolabel(rects)
            plt.savefig("{0}/{1}_{2}.png".format(fig_dir, field, stat))
    return sntype_stats


def autolabel(rects):
    """ Attach a text label above each bar displaying its height """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%d' % int(height), ha='center', va='bottom')


if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    get_distributions_multiprocessing(data_release='20180112', fig_dir=fig_dir)

    plt.show()



