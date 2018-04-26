import sys
import matplotlib.pyplot as plt
import plasticc


def plot_light_curve(data_release='20180407', object_id):
    getter = plasticc.get_data.GetData(data_release)
    field, model, base, snid = objid.astype(str).split('_')
    result = getter.get_lcs_data(field=field, snid=snid, model=model, base=base)
    head, phot = next(result)
    lc = getter.convert_pandas_lc_to_recarray_lc(phot)

    for pb in ['u', 'r', 'i', 'g', 'z', 'Y']:
        plt.errorbar(phot[pb]['MJD'], phot[pb]['FLUXCAL'], yerr=phot[pb]['FLUXCALERR'], fmt='.', label=pb)
    plt.legend()


def main():
    """ Input `python plot_light_curve.py objid data_release` """

    objid = str(sys.argv[1])
    if len(sys.argv > 2):
        data_release = str(sys.argv[2])
    else:
        data_release = '20180407'

    plot_light_curve(data_release, objid)


if __name__ == '__main__':
    sys.exit(main())