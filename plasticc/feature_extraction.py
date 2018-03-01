import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from plasticc.get_data import GetData
from ANTARES_object.LAobject import LAobject


def get_light_curves(data_release, field_in='%', sntype_in='%', snid_in='%', limit=100, shuffle=False):
    getdata = GetData(data_release)
    result = getdata.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd'], field=field_in, 
                                  sntype=sntype_in, snid=snid_in, limit=limit, shuffle=shuffle, sort=False)

    sntypes_map = getdata.get_sntypes()
    sntype_name = sntypes_map[sntype_in]

    non_transients = ['RRLyrae', 'Mdwarf', 'Mira']
    periodic = True if sntype_name in non_transients else False

    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd = head

        for f in phot.columns:  # Filter names
            data = phot.get(f)
            if data is None:
                continue
            flt, flux, fluxerr, mjd, zeropt = data
            t = mjd - peak_mjd

            filtered_err = sigma_clip(fluxerr, sigma=3., iters=5, copy=True)
            filtered_flux = sigma_clip(flux, sigma=7., iters=5, copy=True)
            bad1 = filtered_err.mask
            bad2 = filtered_flux.mask
            ind = ~np.logical_or(bad1, bad2)

            t = t[ind]
            flux = flux[ind]
            fluxerr = fluxerr[ind]
            mjd = mjd[ind]
            flt = flt[ind]

            yield t, flux, fluxerr, zeropt[0], mjd, flt, objid


def get_antares_features():
    """
    Get antares object features.
    Return as a DataFrame with columns being the features, and rows being the objid&passband
    """
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    data_release = '20180112'
    field = 'DDF'
    sntype = 1
    features = {}

    lc_result = get_light_curves(data_release=data_release, field_in=field, sntype_in=sntype, snid_in='%', limit=10, 
                                 shuffle=False)

    for lcinfo in lc_result:
        t, flux, fluxerr, zeropt, mjd, flt, objid = lcinfo
        p = flt[0]
        features[objid] = {}
        laobject = LAobject(locusId=objid, objectId=objid, time=t, flux=flux, fluxErr=fluxerr, obsId=mjd, passband=flt, 
                            zeropoint=zeropt, per=False)

        # Get Features
        features[objid]['amplitude'] = laobject.get_amplitude()
        features[objid]['stats'] = laobject.get_stats()
        features[objid]['skew'] = laobject.get_skew()
        features[objid]['somean'] = laobject.get_StdOverMean()
        features[objid]['shapiro'] = laobject.get_ShapiroWilk()
        features[objid]['q31'] = laobject.get_Q31()
        features[objid]['rms'] = laobject.get_RMS()
        features[objid]['mad'] = laobject.get_MAD()
        features[objid]['stetsonj'] = laobject.get_StetsonJ()
        features[objid]['stetsonk'] = laobject.get_StetsonK()
        features[objid]['acorr'] = laobject.get_AcorrIntegral()
        features[objid]['hlratio'] = laobject.get_hlratio()

        # spline = laobject.spline_smooth(per=False)[flt[0]][0][1].transpose()
        # lc = laobject.get_lc(smoothed=True)[flt[0]]
        # plt.plot(t, flux, label='data')
        # plt.plot(lc[0], lc[1], label='antares lc')
        # plt.plot(spline[0], spline[1], label='spline')
        # plt.legend()
        # break

    features = pd.DataFrame(features)

    print(features)
    return features


if __name__ == '__main__':
    get_antares_features()
    plt.show()
