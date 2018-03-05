import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
import astropy.table as at
import pymysql
from collections import OrderedDict
from plasticc.get_data import GetData
from ANTARES_object.LAobject import LAobject
from . import database


def get_light_curves(data_release, field_in='%', sntype_in='%', snid_in='%', limit=None, shuffle=False):
    getdata = GetData(data_release)
    result = getdata.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd', 'sim_redshift_host'], field=field_in,
                                  sntype=sntype_in, snid=snid_in, limit=limit, shuffle=shuffle, sort=False)

    sntypes_map = getdata.get_sntypes()
    sntype_name = sntypes_map[sntype_in]

    non_transients = ['RRLyrae', 'Mdwarf', 'Mira']
    periodic = True if sntype_name in non_transients else False

    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd, redshift = head

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

            yield t, flux, fluxerr, zeropt[0], mjd, flt, objid, sntype_in, redshift


def save_antares_features():
    """
    Get antares object features.
    Return as a DataFrame with columns being the features, and rows being the objid&passband
    """
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    data_release = '20180221'
    field = 'DDF'
    sntype = 1
    features = {}
    redo = True
    table_exists = False
    nrows = 0
    features_out = []

    lc_result = get_light_curves(data_release=data_release, field_in=field, sntype_in=sntype, snid_in='%', limit=None,
                                 shuffle=True)

    for lcinfo in lc_result:
        t, flux, fluxerr, zeropt, mjd, flt, objid, sntype, redshift = lcinfo
        p = flt[0]
        objid = objid + '_' + p
        features[objid] = OrderedDict()

        try:
            laobject = LAobject(locusId=objid, objectId=objid, time=t, flux=flux, fluxErr=fluxerr, obsId=mjd, passband=flt,
                                zeropoint=zeropt, per=False)
        except ValueError as e:
            print(e)  # No good observations
            continue

        # Get Features
        features[objid]['objid'] = objid
        features[objid]['sntype'] = sntype
        features[objid]['redshift'] = redshift
        stats = laobject.get_stats()[p]
        features[objid]['nobs'] = stats.nobs
        if stats.nobs <= 3:  # Don't store features of light curves with less than 3 points
            continue
        features[objid]['variance'] = stats.variance
        features[objid]['skewness'] = stats.skewness
        features[objid]['kurtosis'] = stats.kurtosis
        features[objid]['min'] = stats.minmax[0]
        features[objid]['max'] = stats.minmax[1]
        features[objid]['amplitude'] = laobject.get_amplitude()[p]
        features[objid]['skew'] = laobject.get_skew()[p]
        features[objid]['somean'] = laobject.get_StdOverMean()[p]
        features[objid]['shapiro'] = laobject.get_ShapiroWilk()[p]
        features[objid]['q31'] = laobject.get_Q31()[p]
        features[objid]['rms'] = laobject.get_RMS()[p]
        features[objid]['mad'] = laobject.get_MAD()[p]
        features[objid]['stetsonj'] = laobject.get_StetsonJ()[p]
        features[objid]['stetsonk'] = laobject.get_StetsonK()[p]
        features[objid]['acorr'] = laobject.get_AcorrIntegral()[p]
        features[objid]['hlratio'] = laobject.get_hlratio()[p]

        if not table_exists:
            if redo:
                database.exec_sql_query("TRUNCATE TABLE features;")  # Delete everything in the features table
            mysql_fields = list(features[objid].keys())
            mysql_formats = ['VARCHAR(255)', ] + ['FLOAT' for x in mysql_fields[1:]]
            mysql_schema = ', '.join(['{} {}'.format(x, y) for x, y in zip(mysql_fields, mysql_formats)])
            table_name = database.create_sql_index_table_for_release(data_release, mysql_schema, redo=redo, table_name='features')
            table_exists = True

        features_out += [list(features[objid].values())]
        nrows += 1

        # Save to mysql in batches of 1000
        if nrows >= 1000:
            features_out = np.array(features_out)
            features_out = at.Table(features_out, names=list(features[objid].keys()))
            features_out = np.array(features_out).tolist()
            nsavedrows = database.write_rows_to_index_table(features_out, table_name)
            print("Saved ", nsavedrows, "rows")
            nrows = 0
            features_out = []

    if features_out:
        features_out = np.array(features_out)
        features_out = at.Table(features_out, names=list(features[objid].keys()))
        features_out = np.array(features_out).tolist()
        nsavedrows = database.write_rows_to_index_table(features_out, table_name)
        print("Saved ", nsavedrows, "rows")

    features = pd.DataFrame(features)

    print(features)
    return features


if __name__ == '__main__':
    save_antares_features()
    plt.show()


