import h5py
import numpy as np


def get_feature_names(passbands, ignore=()):
    feature_names = ['objid', 'redshift']
    feature_names += sum([['variance_%s' % p, 'kurtosis_%s' % p, 'filt-variance_%s' % p, 'filt-kurtosis_%s' % p,
                           'shapiro_%s' % p, 'p-value_%s' % p, 'skew_%s' % p, 'q31_%s' % p,
                           'stetsonk_%s' % p, 'acorr_%s' % p, 'von-neumann_%s' % p, 'hlratio_%s' % p,
                           'amplitude_%s' % p, 'filt-amplitude_%s' % p,  'somean_%s' % p, 'rms_%s' % p, 'mad_%s' % p,
                           'stetsonj_%s' % p, 'stetsonl_%s' % p, 'entropy_%s' % p, 'nobs4096_%s' % p,
                           'risetime_%s' % p, 'riserate_%s' % p] for p in passbands], [])

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


def get_features(fpath, data_release, field_in='%', model_in='%', aggregate_classes=False, helpers=None):
    """ Get features from hdf5 files. """
    hdffile = h5py.File(fpath, 'r')
    features = np.array(hdffile[data_release])
    hdffile.close()
    # features = features[np.random.randint(features.shape[0], size=100000)]

    if aggregate_classes:
        agg_map = helpers.aggregate_sntypes(reverse=True)

    indexes = []
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')

        if aggregate_classes is True:
            submodels = agg_map[int(model_in)]
            for m in submodels:
                if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(m)):
                    indexes.append(i)
        else:
            if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(model_in)):
                indexes.append(i)

    features = features[indexes]

    return features