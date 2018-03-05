import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import astropy.table as at
import pymysql
from collections import OrderedDict
from plasticc.get_data import GetData
from ANTARES_object.LAobject import LAobject
from . import database

DIRNAMES = 1


def get_light_curves(data_release, field_in='%', sntype_in='%', snid_in='%', limit=None, shuffle=False):
    getdata = GetData(data_release)
    result = getdata.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd', 'sim_redshift_host'], field=field_in,
                                  sntype=sntype_in, snid=snid_in, limit=limit, shuffle=shuffle, sort=False)

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

        yield t, flux, fluxerr, zeropt, mjd, flt, objid, sntype_in, redshift


def save_antares_features(data_release, redo=False):
    """
    Get antares object features.
    Return as a DataFrame with columns being the features, and rows being the objid&passband
    """

    field = 'DDF'
    sntype = '1'
    features = {}
    table_exists = False
    nrows = 0
    features_out = []
    feature_fields = sum([['variance_%s' % pb, 'kurtosis_%s' % pb, 'amplitude_%s' % pb, 'skew_%s' % pb, 'somean_%s' % pb,
                           'shapiro_%s' % pb, 'q31_%s' % pb, 'rms_%s' % pb, 'mad_%s' % pb, 'stetsonj_%s' % pb,
                           'stetsonk_%s' % pb, 'acorr_%s' % pb, 'hlratio_%s' % pb] for pb in ['i', 'r', 'Y', 'u', 'g', 'z']], [])
    mysql_fields = ['objid', 'sntype', 'redshift'] + feature_fields

    lc_result = get_light_curves(data_release=data_release, field_in=field, sntype_in=sntype, snid_in='%', limit=None,
                                 shuffle=True)

    for lcinfo in lc_result:
        t, flux, fluxerr, zeropt, mjd, flt, objid, sntype, redshift = lcinfo
        features = OrderedDict()
        features['objid'] = objid
        features['sntype'] = sntype
        features['redshift'] = redshift
        for pb in ['i', 'r', 'Y', 'u', 'g', 'z']:
            try:
                laobject = LAobject(locusId=objid, objectId=objid, time=t[pb], flux=flux[pb], fluxErr=fluxerr[pb], obsId=mjd[pb], passband=flt[pb],
                                    zeropoint=zeropt[pb], per=False)
            except (ValueError, KeyError) as e:
                print(e)  # No good observations
                for key in feature_fields:
                    features[key] = 999
                continue
            # Get Features
            stats = laobject.get_stats()[pb]
            if stats.nobs <= 3:  # Don't store features of light curves with less than 3 points
                continue
            features['variance_%s' % pb] = stats.variance
            features['kurtosis_%s' % pb] = stats.kurtosis
            features['amplitude_%s' % pb] = laobject.get_amplitude()[pb]
            features['skew_%s' % pb] = laobject.get_skew()[pb]
            features['somean_%s' % pb] = laobject.get_StdOverMean()[pb]
            features['shapiro_%s' % pb] = laobject.get_ShapiroWilk()[pb]
            features['q31_%s' % pb] = laobject.get_Q31()[pb]
            features['rms_%s' % pb] = laobject.get_RMS()[pb]
            features['mad_%s' % pb] = laobject.get_MAD()[pb]
            features['stetsonj_%s' % pb] = laobject.get_StetsonJ()[pb]
            features['stetsonk_%s' % pb] = laobject.get_StetsonK()[pb]
            features['acorr_%s' % pb] = laobject.get_AcorrIntegral()[pb]
            features['hlratio_%s' % pb] = laobject.get_hlratio()[pb]

        if not table_exists:
            if redo:
                database.exec_sql_query("TRUNCATE TABLE features_{};".format(data_release))  # Delete everything in the features table
            # mysql_fields = [field.lower() for field in mysql_fields]
            mysql_formats = ['VARCHAR(255)', ] + ['FLOAT' for x in mysql_fields[1:]]
            mysql_schema = ', '.join(['{} {}'.format(x, y) for x, y in zip(mysql_fields, mysql_formats)])
            table_name = database.create_sql_index_table_for_release(data_release, mysql_schema, redo=redo, table_name='features_{}'.format(data_release))
            table_exists = True

        features_out += [list(features.values())]
        nrows += 1

        # Save to mysql in batches of 1000
        if nrows >= 1000:
            features_out = np.array(features_out)
            features_out = at.Table(features_out, names=mysql_fields)
            features_out = np.array(features_out).tolist()
            nsavedrows = database.write_rows_to_index_table(features_out, table_name)
            print("Saved ", nsavedrows, "rows")
            nrows = 0
            features_out = []

    if features_out:
        features_out = np.array(features_out)
        features_out = at.Table(features_out, names=mysql_fields)
        features_out = np.array(features_out).tolist()
        nsavedrows = database.write_rows_to_index_table(features_out, table_name)
        print("Saved ", nsavedrows, "rows")

    # features = pd.DataFrame(features)

    # print(features)
    return features


if __name__ == '__main__':
    # data_dir = os.path.join(ROOT_DIR, 'plasticc_data')
    # for data_release in next(os.walk(data_dir))[DIRNAMES]:
    #     if data_release == 'src':
    #         continue

    save_antares_features(data_release='20180221', redo=True)

