import sys
import matplotlib.pyplot as plt
import plasticc.get_data


def plot_light_curve(objid, data_release='20180407'):
    getter = plasticc.get_data.GetData(data_release)
    field, model, base, snid = objid.split('_')
    result = getter.get_lcs_data(field=field, snid=snid, model=model, base=base)
    head, phot = next(result)
    objid, ptrobs_min, ptrobs_max = head
    lc = getter.convert_pandas_lc_to_recarray_lc(phot)
    col = {'u': 'b', 'g': 'g', 'r': 'r', 'i': 'm', 'z': 'k', 'Y': 'y'}
    for pb in phot.keys():
        plt.errorbar(phot[pb]['MJD'], phot[pb]['FLUXCAL'], yerr=phot[pb]['FLUXCALERR'], fmt='.', label=pb, color=col[pb])
    plt.legend()

    plt.title(objid)
    plt.xlabel('mjd')
    plt.ylabel('flux')
    plt.savefig('temp_light_curve')


def main():
    """
    Usage:
        python plot_light_curve.py objid data_release

    Example:
         python -m bin.plot_light_curve WFD_80_%_%
    """

    objid = str(sys.argv[1])
    if len(sys.argv) > 2:
        data_release = str(sys.argv[2])
    else:
        data_release = '20180407'

    print(objid, data_release)

    plot_light_curve(objid, data_release)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())