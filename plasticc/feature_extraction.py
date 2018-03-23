import os
import sys
ROOT_DIR = os.getenv('PLASTICC_DIR')
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
import astropy.table as at
from collections import OrderedDict
from plasticc.get_data import GetData
from ANTARES_object.LAobject import LAobject
import h5py
import multiprocessing as mp
import extinction
from . import database

DIRNAMES = 1


def set_keys_to_nan(feature_fields, p, features):
    for key in feature_fields:
        if '_%s' % p in key:
            features[key] = np.nan
    return features


def renorm_flux_lightcurve(flux, fluxerr, mu):
    """ Normalise flux light curves with distance modulus."""
    d = 10 ** (mu/5 + 1)
    dsquared = d**2

    norm = 1e19

    fluxout = flux * dsquared / norm
    fluxerrout = fluxerr * dsquared / norm

    return fluxout, fluxerrout


def remove_extinction(mwebv, lc):
    passbands = ['u', 'g', 'r', 'i', 'z', 'Y']
    PB_WAVE = np.array([3569.5, 4766.5, 6214.5, 7544.5, 8707.5, 10039.5])

    # Using negative a_v so that extinction.apply works in reverse and removes the extinction
    extinctions = extinction.fitzpatrick99(wave=PB_WAVE, a_v=-3.1*mwebv, r_v=3.1, unit='aa')

    for i, pb in enumerate(passbands):
        flux = lc['flux'][lc['pb'] == pb]
        fluxerr = lc['dflux'][lc['pb'] == pb]

        extinction.apply(extinctions[i], flux)
        newflux = extinction.apply(extinctions[i], flux, inplace=False)
        newfluxerr = extinction.apply(extinctions[i], fluxerr, inplace=False)

        lc['flux'][lc['pb'] == pb] = newflux
        lc['dflux'][lc['pb'] == pb] = newfluxerr

    return lc


def save_antares_features(data_release, fname, field_in='%', model_in='%', batch_size=100, offset=0, sort=True, redo=False):
    """
    Get antares object features.
    Return as a DataFrame with columns being the features, and rows being the objid&passband
    """
    print(fname)
    passbands = ['u', 'g', 'r', 'i', 'z', 'Y']
    features_out = []
    feature_fields = sum([['variance_%s' % p, 'kurtosis_%s' % p, 'amplitude_%s' % p, 'skew_%s' % p, 'somean_%s' % p,
                           'shapiro_%s' % p, 'q31_%s' % p, 'rms_%s' % p, 'mad_%s' % p, 'entropy_%s' % p,
                           'stetsonj_%s' % p, 'stetsonk_%s' % p, 'acorr_%s' % p, 'von-neumann_%s' % p,
                           'hlratio_%s' % p] for p in passbands], [])

    mysql_fields = ['objid', 'redshift'] + feature_fields

    getter = GetData(data_release)
    result = getter.get_lcs_data(columns=['objid', 'ptrobs_min', 'ptrobs_max', 'peakmjd', 'sim_redshift_host', 'mwebv', 'sim_dlmu'], field=field_in,
                                  model=model_in, snid='%', limit=batch_size, offset=offset, shuffle=False, sort=sort)
    count = 0
    for head, phot in result:
        objid, ptrobs_min, ptrobs_max, peak_mjd, redshift, mwebv, dlmu = head
        lc = getter.convert_pandas_lc_to_recarray_lc(phot)
        lc = remove_extinction(mwebv, lc)

        obsid = np.arange(len(lc))
        t = lc['mjd'] - peak_mjd  # subtract peakmjd from each mjd.
        flux, fluxerr = renorm_flux_lightcurve(flux=lc['flux'], fluxerr=lc['dflux'], mu=dlmu)
        try:
            laobject = LAobject(locusId=objid, objectId=objid, time=t, flux=flux, fluxErr=fluxerr,
                                obsId=obsid, passband=lc['pb'], zeropoint=lc['zpt'], per=False, mag=False)
        except ValueError as err:
            print(err)
            continue
        features = OrderedDict()
        features['objid'] = objid.encode('utf8')
        features['redshift'] = redshift

        for p in passbands:
            try:
                stats = laobject.get_stats()[p]
                if stats.nobs <= 3:  # Don't store features of light curves with less than 3 points
                    features = set_keys_to_nan(feature_fields, p, features)
                    continue
                features['variance_%s' % p] = stats.variance
                features['kurtosis_%s' % p] = stats.kurtosis
                try:
                    features['amplitude_%s' % p] = laobject.get_amplitude()[p]
                except AttributeError as err:
                    # TODO: AttributeError is caused by not enough nobs in get_amplitude function in the ANTARES object. Look into why this differs from stats.nobs
                    # print("Attribute error for {}, {}-band: {}".format(objid, p, err))
                    features['amplitude_%s' % p] = np.nan

                features['skew_%s' % p] = laobject.get_skew()[p]
                features['somean_%s' % p] = laobject.get_StdOverMean()[p]
                features['shapiro_%s' % p] = laobject.get_ShapiroWilk()[p]
                features['q31_%s' % p] = laobject.get_Q31()[p]
                features['rms_%s' % p] = laobject.get_RMS()[p]
                features['mad_%s' % p] = laobject.get_MAD()[p]
                features['entropy_%s' % p] = laobject.get_ShannonEntropy()[p]
                features['stetsonj_%s' % p] = laobject.get_StetsonJ()[p]
                features['stetsonk_%s' % p] = laobject.get_StetsonK()[p]
                features['acorr_%s' % p] = laobject.get_AcorrIntegral()[p]
                features['von-neumann_%s' % p] = laobject.get_vonNeumannRatio()[p]
                features['hlratio_%s' % p] = laobject.get_hlratio()[p]
            except KeyError as err:
                features = set_keys_to_nan(feature_fields, p, features)
                print('NO FEATURES FOR: ', objid, p)
                continue
        count += 1
        # field, model, base, snid = objid.astype(str).split('_')
        print(objid.encode('utf8'), offset, count, os.path.basename(fname))
        features_out += [list(features.values())]

    # print('__B__', offset, count, len(features_out), os.path.basename(fname))
    # Set all columns to floats except set first column to string (objid)
    dtypes = ['S24'] + [np.float64] * (len(mysql_fields) - 1)

    # Save to hdf5 in batches of 10000
    features_out = np.array(features_out)
    features_out = at.Table(features_out, names=mysql_fields, dtype=dtypes)
    features_out.write(fname, path=data_release, append=False, overwrite=redo)
    print("saved %s" % fname)

    return fname


def combine_hdf_files(save_dir, data_release):
    fnames = os.listdir(save_dir)
    fname_out = os.path.join(ROOT_DIR, 'plasticc', 'features_all_DDF.hdf5')
    output_file = h5py.File(fname_out, 'w')

    # keep track of the total number of rows
    total_rows = 0

    for n, f in enumerate(fnames):
        f_hdf = h5py.File(os.path.join(save_dir, f), 'r')
        data = f_hdf[data_release]
        total_rows = total_rows + data.shape[0]

        if n == 0:
            # first file; fill the first section of the dataset; create with no max shape
            create_dataset = output_file.create_dataset(data_release, data=data, chunks=True, maxshape=(None,))
            where_to_start_appending = total_rows
        else:
            # resize the dataset to accomodate the new data
            create_dataset.resize(total_rows, axis=0)
            create_dataset[where_to_start_appending:total_rows] = data
            where_to_start_appending = total_rows

        f_hdf.close()

    output_file.close()


def create_all_hdf_files(data_release, i, save_dir, field_in, model_in, batch_size, sort, redo):
    offset = batch_size * i
    fname = os.path.join(save_dir, 'features_{}.hdf5'.format(i))
    save_antares_features(data_release=data_release, fname=fname, field_in=field_in, model_in=model_in,
                          batch_size=batch_size, offset=offset, sort=sort, redo=redo)

def main():
    save_dir = os.path.join(ROOT_DIR, 'plasticc', 'hdf_features')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # data_dir = os.path.join(ROOT_DIR, 'plasticc_data')
    # for data_release in next(os.walk(data_dir))[DIRNAMES]:
    #     if data_release == 'src':
    #         continue
    #     save_antares_features(data_release, redo=True)

    data_release = '20180316'
    field = 'DDF'
    model = '%'
    getter = GetData(data_release)
    nobjects = next(getter.get_lcs_headers(field=field, model=model, get_num_lightcurves=True, big=False))
    print("{} objects for model {} in field {}".format(nobjects, model, field))

    batch_size = 2000
    sort = True
    redo = True

    # offset = 0
    # i = 0
    # while offset < 1000:
    #     fname = os.path.join(save_dir, 'features_{}.hdf5'.format(i))
    #     save_antares_features(data_release=data_release, fname=fname, field_in=field, model_in=model,
    #                           batch_size=batch_size, offset=offset, sort=sort, redo=redo)
    #     offset += batch_size
    #     i += 1

    # Multiprocessing
    i_list = np.arange(0, int(nobjects/batch_size) + 1)
    print(i_list)
    pool = mp.Pool()
    results = [pool.apply_async(create_all_hdf_files, args=(data_release, i, save_dir, field, model, batch_size, sort, redo)) for i in i_list]
    print(results)
    pool.close()
    pool.join()

    # The last file with less than the batch_size number of objects isn't getting saved. If so, retry saving it here:
    fname_last = os.path.join(save_dir, 'features_{}.hdf5'.format(i_list[-1]))
    print(fname_last)
    if not os.path.isfile(fname_last):
        print("Last file not saved. Retrying...")
        save_antares_features(data_release=data_release, fname=fname_last, field_in=field, model_in=model,
                              batch_size=batch_size, offset=batch_size*i_list[-1], sort=sort, redo=redo)

    combine_hdf_files(save_dir, data_release)


if __name__ == '__main__':
    main()





