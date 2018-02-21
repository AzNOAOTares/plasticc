import os
import numpy as np
import matplotlib.pyplot as plt
from get_data import GetData

ROOT_DIR = os.getenv('PLASTICC_DIR')


def plot_light_curves(data_release, fig_dir=None, field_in='%', sntype_in='%', snid_in='%', cadences=None):
    getdata = GetData(data_release)
    result = getdata.get_transient_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd'], field=field_in, sntype=sntype_in, snid=snid_in)
    t_plot, mag_plot, magerr_plot, peak_mjd_plot = {}, {}, {}, {}
    sntypes, sntypes_map = getdata.get_sntypes()
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
            flt, mag, magerr, mjd = phot[f]
            g = np.where(mag != 128.0)[0]  # good indexes (where magnitude isn't 128)
            flt, mag, magerr, mjd = flt[g], mag[g], magerr[g], mjd[g]
            # print(flt, mjd, mag, magerr)

            cad = np.median(np.diff(mjd))
            if sntype_name in non_transients or  min(peak_mjd_all) + 90 < peak_mjd < max(peak_mjd_all) - 90:
                if 0.3 * med_cad < cad < med_cad*1.2:
                    t = mjd - peak_mjd
                    if f in t_plot:
                        if n[f] < 6:
                            n[f] += 1
                            t_plot[f].append(t)
                            mag_plot[f].append(mag)
                            magerr_plot[f].append(magerr)
                            peak_mjd_plot[f].append(peak_mjd)
                            print("Plotted", n, f, cad)

                    else:
                        t_plot[f], mag_plot[f], magerr_plot[f], peak_mjd_plot[f] = [], [], [], []
                else:
                    print("CADENCE TOO HIGH/LOW: ", cad, f, sntype_name)
            else:
                print("OUT OF SEASON:", n, peak_mjd, sntype_name, min(peak_mjd_all) + 90, max(peak_mjd_all) - 90)


    # try:
    if peak_mjd_all != []:  # If there are nonzero light curves
        fig = plt.figure(figsize=(15, 10))
        for i, f in enumerate(t_plot.keys()):
            ax = fig.add_subplot(3, 2, i + 1)
            fig.tight_layout()
            fig.suptitle("{}".format(sntypes_map[sntype_in]))
            ax.set_title(f)
            ax.invert_yaxis()
            if f not in t_plot.keys():
                continue
            for i in range(len(t_plot[f])):
                ax.plot(t_plot[f][i], mag_plot[f][i], marker='.')
            fig.savefig("{0}/{1}_{2}_{3}_{4}.png".format(fig_dir, field_in, sntype_in, snid_in, f))
        # except Exception as e:
        #     print("No results for {}_{}_{}.png".format(field_in, sntype_in, snid_in))


    # plt.plot(t, mag, marker='.', label="{}".format(snid))
    # # plt.errorbar(mjd, mag, yerr=magerr, fmt='.')
    # plt.gca().invert_yaxis()
    # plt.title("{}: {}".format(sntypes_map[sntype], f))
    # plt.xlabel('Days since date of min(mag)')
    # plt.ylabel('Mag')
    # plt.legend()

if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    print("PLOTTING LIGHTCURVES")

    epoch_ranges = {'SN1a': 112.5, 'CC': 112.1, 'SNIbc': 110.9, 'IIn': 115.6, 'SNIa-91bg': 111.0, 'pointIa': 106.1,
                    'Kilonova': 0., 'Magnetar': 560.9, 'PISN': 252.5, 'ILOT': 116.6, 'CART': 488.6,
                    'RRLyrae': 818.4, 'Mdwarf': 804.5}

    median_cadences = {'SN1a': 7.279, 'CC': 7.525, 'SNIbc': 7.621, 'IIn': 7.102, 'SNIa-91bg': 7.855, 'pointIa': 8.835,
                    'Kilonova': 0., 'Magnetar': 7.145, 'PISN': 7.256, 'ILOT': 6.987, 'CART': 8.502,
                    'RRLyrae': 18.37, 'Mdwarf': 21.79, 'Mira': 7., 'BSR': 7., 'String': 7}

    for s in [1, 2, 3, 4, 42, 45, 50, 60, 61, 62, 63, 80, 81, 82]:
        plot_light_curves(data_release='20180112', fig_dir=fig_dir, field_in='DDF', sntype_in=s, snid_in='%', cadences=median_cadences)
    print("PLOTTED LIGHTCURVES")

    plt.show()


