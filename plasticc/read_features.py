import h5py
import numpy as np

import helpers


def get_feature_names(passbands, ignore=()):
    feature_names = ['objid', 'redshift']
    feature_names += sum([['variance_%s' % p, 'kurtosis_%s' % p, 'filt-kurtosis_%s' % p,
                          'shapiro_%s' % p, 'p-value_%s' % p, 'skew_%s' % p, 'q31_%s' % p,
                          'stetsonk_%s' % p, 'acorr_%s' % p, 'von-neumann_%s' % p, 'hlratio_%s' % p,
                          'amplitude_%s' % p, 'filt-amplitude_%s' % p, 'somean_%s' % p, 'rms_%s' % p, 'mad_%s' % p,
                          'stetsonj_%s' % p, 'stetsonl_%s' % p, 'entropy_%s' % p] for p in passbands], [])
    color_fields = []
    for i, pb1 in enumerate(passbands):
        for j, pb2 in enumerate(passbands):
            if i < j:
                color = pb1 + '-' + pb2
                color_fields += ['amp %s' % color]
                color_fields += ['mean %s' % color]
    feature_names += color_fields
    period_fields = ['period1', 'period_score1', 'period2', 'period_score2', 'period3', 'period_score3', 'period4',
                     'period_score4', 'period5', 'period_score5']
    feature_names += period_fields

    for name in ignore:
        feature_names.remove(name)

    feature_names = np.array(feature_names)

    return feature_names


def get_features(fpath, data_release, field_in='%', model_in='%', aggregate_classes=False):
    """ Get features from hdf5 files. """
    hdffile = h5py.File(fpath, 'r')
    features = np.array(hdffile[data_release])
    hdffile.close()
    if aggregate_classes:
        agg_map = helpers.aggregate_sntypes(reverse=True)

    indexes = []
    modelList = []
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        modelList.append(model)

        if aggregate_classes is True:
            submodels = agg_map[int(model_in)]
            for m in submodels:
                if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(m)):
                    indexes.append(i)
        else:
            if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(model_in)):
                indexes.append(i)

    features = features[indexes]

    # rescaled_flux_new = []
    # rescaled_flux = features['rescaled-flux_r']
    # for s in rescaled_flux:
    #     s = s.decode('utf-8').replace('[', '')
    #     s = s.replace(']', '')
    #     s = s.replace('\\n', '')
    #     s = np.array(s.split()).astype(float)
    #     # rescaled_flux_new.append(s)
    #     rescaled_flux_new += list(s)
    # rescaled_flux_new = np.asarray(rescaled_flux_new)
    #
    # fig = plt.figure()
    #
    # num_bins = 50
    # # the histogram of the data
    # # n, bins, patches = plt.hist(rescaled_flux_new, num_bins, normed=1, facecolor='green', alpha=0.5)
    # sns.kdeplot(rescaled_flux_new)
    # plt.xlabel('rescaled-flux')
    # plt.title(model_in)
    # plt.show()

    return features