import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from plot_features import get_features
import helpers

ROOT_DIR = os.getenv('PLASTICC_DIR')


def get_labels_and_features(fpath, data_release, field, model, feature_names, pb):
    X = []  # features
    y = []  # labels

    features = get_features(fpath, data_release, field, model)
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        y.append(int(model))
        x = []
        for f in feature_names:
            x.append(features[i]["%s_%s" % (f, pb)])
        x = np.array(x)
        X.append(x)

    X = np.array(X)
    y = np.array(y)

    # Remove rows that contain any NaNs in them
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    # Remove extreme values over 10 standard deviations from the median 10 times iteratively
    for ii in range(10):
        for f in range(X.shape[1]):
            std = np.std(X[:, f])
            median = np.median(X[:, f])
            mask = np.where(abs(X[:, f] - median) < 10 * std)[0]
            if np.where(abs(X[:, f] - median) > 10 * std)[0].any():
                pass
            X = X[mask]
            y = y[mask]

    return X, y


def classify(X, y, models, sntypes_map, feature_names):
    # Split into train/test
    XTrain, XTest, yTrain, yTest = train_test_split(X, y)

    # Train model
    model = RandomForestClassifier()
    model.fit(XTrain, yTrain)

    # visualize test performance
    yPred = model.predict(XTest)

    accuracy = len(np.where(y==yPred)[0])
    print("Accuracy is: {}".format(accuracy))

    colors = ('r', 'b', 'g', 'm', 'c')
    fig, ax = plt.subplots(nrows=len(feature_names), ncols=len(feature_names), sharex='col', sharey='row', figsize=(18,15))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i, feat1 in enumerate(feature_names):
        for j, feat2 in enumerate(feature_names):
            for model, color in zip(models, colors):
                model_name = sntypes_map[model]
                ax[i, j].scatter(XTrain[:, j][yTrain == model], XTrain[:, i][yTrain == model], color=color, marker='.', alpha=0.3, label="Train_%s" % model_name)
                ax[i, j].scatter(XTest[:, j][yTest == model], XTest[:, i][yTest == model], color=color, marker='.', label="Test_%s" % model_name)
                if i == len(feature_names) - 1:
                    ax[i, j].set_xlabel(feat2)
                if j == 0:
                    ax[i, j].set_ylabel(feat1)
    ax[0, j].legend(loc='upper left', bbox_to_anchor=(1,1))


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'classify')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_all_DDF.hdf5')
    sntypes_map = helpers.get_sntypes()

    data_release = '20180221'
    field = 'DDF'
    model = '%'

    feature_names = ('variance', 'kurtosis', 'amplitude', 'skew', 'somean', 'shapiro', 'q31', 'rms', 'mad', 'stetsonj', 'stetsonk', 'acorr', 'hlratio')
    # feature_names = ('variance', 'kurtosis', 'amplitude', 'skew')

    X, y = get_labels_and_features(fpath, data_release, field, model, feature_names, 'r')

    models = [1, 2, 3, 42, 45, 60, 61, 62, 63]
    classify(X, y, models, sntypes_map, feature_names)

    plt.show()


if __name__ == '__main__':
    main()
