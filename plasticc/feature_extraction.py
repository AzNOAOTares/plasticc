import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import astropy.table as at
from collections import OrderedDict
from plasticc.get_data import GetData
from ANTARES_object.LAobject import LAobject
import h5py
from . import database

DIRNAMES = 1


def get_light_curves(data_release, field_in='%', model_in='%', snid_in='%', limit=None, shuffle=False):
    getdata = GetData(data_release)
    result = getdata.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd', 'sim_redshift_host'], field=field_in,
                                  model=model_in, snid=snid_in, limit=limit, shuffle=shuffle, sort=False)

    # sntypes_map = getdata.get_sntypes()
    # sntype_name = sntypes_map[sntype_in]
    #
    # non_transients = ['RRLyrae', 'Mdwarf', 'Mira']
    # periodic = True if sntype_name in non_transients else False
    t, flux, fluxerr, zeropt, mjd, flt = {}, {}, {}, {}, {}, {}

    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd, redshift = head

        for pb in phot.columns:  # Filter names
            data = phot.get(pb)
            if data is None:
                continue
            flt[pb], flux[pb], fluxerr[pb], mjd[pb], zeropt[pb] = data
            zeropt[pb] = zeropt[pb][0]
            t[pb] = mjd[pb] - peak_mjd

            filtered_err = sigma_clip(fluxerr[pb], sigma=3., iters=5, copy=True)
            filtered_flux = sigma_clip(flux[pb], sigma=7., iters=5, copy=True)
            bad1 = filtered_err.mask
            bad2 = filtered_flux.mask
            ind = ~np.logical_or(bad1, bad2)

            t[pb] = t[pb][ind]
            flux[pb] = flux[pb][ind]
            fluxerr[pb] = fluxerr[pb][ind]
            mjd[pb] = mjd[pb][ind]
            flt[pb] = flt[pb][ind]

        yield t, flux, fluxerr, zeropt, mjd, flt, objid, model_in, redshift


def save_antares_features(data_release, redo=False):
    """
    Get antares object features.
    Return as a DataFrame with columns being the features, and rows being the objid&passband
    """
    passbands = ['i', 'r', 'Y', 'u', 'g', 'z']
    field = 'DDF'
    model = '1'
    # fname = 'features_{}.hdf5'.format(model)
    fname = os.path.join(ROOT_DIR, 'plasticc', 'features.hdf5')
    nrows = 0
    features_out = []
    feature_fields = sum([['variance_%s' % p, 'kurtosis_%s' % p, 'amplitude_%s' % p, 'skew_%s' % p, 'somean_%s' % p,
                           'shapiro_%s' % p, 'q31_%s' % p, 'rms_%s' % p, 'mad_%s' % p, 'stetsonj_%s' % p,
                           'stetsonk_%s' % p, 'acorr_%s' % p, 'hlratio_%s' % p] for p in passbands], [])

    mysql_fields = ['objid', 'redshift'] + feature_fields

    lc_result = get_light_curves(data_release=data_release, field_in=field, model_in=model, snid_in='%', limit=None,
                                 shuffle=True)

    for lcinfo in lc_result:
        t, flux, fluxerr, zeropt, mjd, flt, objid, model, redshift = lcinfo
        features = OrderedDict()
        features['objid'] = objid.encode('utf8')
        features['redshift'] = redshift
        for p in passbands:
            try:
                laobject = LAobject(locusId=objid, objectId=objid, time=t[p], flux=flux[p], fluxErr=fluxerr[p], obsId=mjd[p], passband=flt[p],
                                    zeropoint=zeropt[p], per=False)
            except (ValueError, KeyError) as e:
                print(e)  # No good observations
                for key in feature_fields:
                    if '_%s' % p in key:
                        features[key] = np.nan
                continue
            # Get Features
            stats = laobject.get_stats()[p]
            if stats.nobs <= 3:  # Don't store features of light curves with less than 3 points
                for key in feature_fields:
                    if '_%s' % p in key:
                        features[key] = np.nan
                continue
            features['variance_%s' % p] = stats.variance
            features['kurtosis_%s' % p] = stats.kurtosis
            features['amplitude_%s' % p] = laobject.get_amplitude()[p]
            features['skew_%s' % p] = laobject.get_skew()[p]
            features['somean_%s' % p] = laobject.get_StdOverMean()[p]
            features['shapiro_%s' % p] = laobject.get_ShapiroWilk()[p]
            features['q31_%s' % p] = laobject.get_Q31()[p]
            features['rms_%s' % p] = laobject.get_RMS()[p]
            features['mad_%s' % p] = laobject.get_MAD()[p]
            features['stetsonj_%s' % p] = laobject.get_StetsonJ()[p]
            features['stetsonk_%s' % p] = laobject.get_StetsonK()[p]
            features['acorr_%s' % p] = laobject.get_AcorrIntegral()[p]
            features['hlratio_%s' % p] = laobject.get_hlratio()[p]
        features_out += [list(features.values())]
        nrows += 1

        # Save to hdf5 in batches of 1000
        if nrows >= 100:
            print(len(features_out[0]), len(mysql_fields))
            features_out = np.array(features_out)
            features_out = at.Table(features_out, names=mysql_fields)
            features_out.write(fname, path=data_release + '/' + model, append=True)
            print("saved")
            nrows = 0
            features_out = []



if __name__ == '__main__':
    # data_dir = os.path.join(ROOT_DIR, 'plasticc_data')
    # for data_release in next(os.walk(data_dir))[DIRNAMES]:
    #     if data_release == 'src':
    #         continue

    save_antares_features(data_release='20180221', redo=True)

