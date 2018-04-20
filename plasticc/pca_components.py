import os
import numpy as np
import h5py
from sklearn.decomposition import PCA, FastICA
from chainconsumer import ChainConsumer

import helpers

from read_features import get_feature_names, get_features

ROOT_DIR = os.getenv('PLASTICC_DIR')


def get_pca_features(features, n_comps=5, feature_names=()):
    X = features[feature_names].view(np.float64).reshape(features[feature_names].shape + (-1,))
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    features = features[mask]

    pca = PCA(n_components=n_comps)
    pca.fit(X)
    weights = pca.fit_transform(X)
    comps = pca.components_.transpose()

    pca_features = np.core.records.fromarrays(np.insert(weights.transpose().astype(object), 0, features['objid'], axis=0),
                                              names=['objid'] + ['comp{}'.format(i+1) for i in range(weights.shape[1])],
                                              formats = ['S24'] + ['f8' for i in range(weights.shape[1])])

    return pca_features


def main():
    fig_dir = os.path.join(ROOT_DIR, 'Figures', 'classify')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'features_test.hdf5')
    sntypes_map = helpers.get_sntypes()
    data_release = '20180407'
    passbands = ('r', 'i', 'z', 'Y')

    features = get_features(fpath, data_release, field_in='DDF', model_in='%', aggregate_classes=True)
    feature_names = get_feature_names(passbands, ignore=('objid',))

    pca_features = get_pca_features(features, n_comps=2, feature_names=feature_names)


if __name__ == '__main__':
    main()
