import os
import numpy as np
import matplotlib.pyplot as plt
from get_data import GetData

ROOT_DIR = os.getenv('PLASTICC_DIR')


def get_class_distributions(data_release, fig_dir):
    getdata = GetData(data_release)
    sntypes = getdata.get_sntypes()
    sntype_stats = {'nobjects': {'DDF': {}, 'WFD': {}}, 'mean_mwebv': {'DDF': {}, 'WFD': {}}, 
                    'mean_epoch_range': {'DDF': {}, 'WFD': {}}, 'mean_cadence': {'DDF': {}, 'WFD': {}}}

    # Get population stats for each field and sntype
    for field in ['DDF', 'WFD']:
        for sntype in sntypes:
            # Get the number of objects for each sntype
            result = getdata.get_transient_data(field=field, sntype=sntype, get_num_lightcurves=True)
            sntype_stats['nobjects'][field][sntype] = next(result)

            n, sum_mwebv, sum_epoch_range, sum_cadence = 0, 0, 0, 0
            result = getdata.get_transient_data(field=field, sntype=sntype, get_num_lightcurves=False)
            for head, phot in result:
                objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = head
                mjd, flt, mag, magerr = phot

                n += 1
                sum_mwebv += mwebv
                sum_epoch_range += (mjd.max() - mjd.min())
                sum_cadence += np.median(np.diff(mjd))

            sntype_stats['mean_mwebv'][field][sntype] = sum_mwebv/n
            sntype_stats['mean_epoch_range'][field][sntype] = sum_epoch_range / n
            sntype_stats['mean_cadence'][field][sntype] = sum_cadence / n

        # Plot the histogram for each statistic
        for stat in sntype_stats:
            plt.bar(range(len(sntypes)), sntype_stats[stat][field].values(), align='center')
            plt.xticks(range(len(sntypes)), sntypes)
            plt.xlabel('sntype')
            plt.ylabel(stat)
            plt.savefig("{0}/{1}_{2}.png".format(fig_dir, field, stat))

    return sntype_stats


if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    get_class_distributions(data_release='20180112', fig_dir=fig_dir)
    plt.show()

