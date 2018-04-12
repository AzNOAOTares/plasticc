import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from plot_features import get_features
import helpers
from chainconsumer import ChainConsumer

ROOT_DIR = '..'  # os.getenv('PLASTICC_DIR')


def get_labels_and_features(fpath, data_release, field, model, feature_names, passbands):
    X = []  # features
    y = []  # labels

    features = get_features(fpath, data_release, field, model)
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        y.append(int(model))
        x = []
        # for f in feature_names:
        #     x.append(features[i]["%s_%s" % (f, pb)])
        for f in feature_names:
            x.append(features[i][f])
        x = np.array(x)
        X.append(x)

    X = np.array(X)
    y = np.array(y)

    print("Num objects before removing objects where any features are NaN: ", len(X))
    # Remove rows that contain any NaNs in them
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    print("Num objects after removing objects where any features are NaN: ", len(X))

    # Remove extreme values over 20 standard deviations from the median 10 times iteratively
    for ii in range(1):
        for f in range(X.shape[1]):
            std = np.std(X[:, f])
            median = np.median(X[:, f])
            mask = np.where(abs(X[:, f] - median) < 20 * std)[0]
            if np.where(abs(X[:, f] - median) > 20 * std)[0].any():
                pass
            X = X[mask]
            y = y[mask]

    return X, y


def classify(X, y, models, sntypes_map, feature_names, fig_dir='.'):
    # Remove models before training:
    mask = np.where(y != 1)[0]
    X = X[mask]
    y = y[mask]
    mask = np.where(y != 2)[0]
    X = X[mask]
    y = y[mask]

    # Split into train/test
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, train_size=0.75, shuffle=True)

    # Train model
    model_ml = RandomForestClassifier(n_estimators=200)
    model_ml.fit(XTrain, yTrain)

    importances = model_ml.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_ml.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]


    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure(figsize=(15, 8))
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance'))
    plt.show()

    # visualize test performance
    yPred = model_ml.predict(XTest)

    accuracy = len(np.where(yPred==yTest)[0])
    print("Accuracy is: {}/{} = {}".format(accuracy, len(yPred), accuracy/len(yPred)))

    # colors = ('#e6194b', '#0082c8', '#3cb44b', '#f032e6', '#46f0f0', '#ffe119', '#f58231', '#911eb4', '#d2f53c', '#008080', '#fabebe', '#e6beff', '#aa6e28', '#000080', '#000000')
    # fig, ax = plt.subplots(nrows=len(feature_names), ncols=len(feature_names), sharex='col', sharey='row', figsize=(18,15))
    # fig.subplots_adjust(wspace=0, hspace=0)
    # for i, feat1 in enumerate(feature_names):
    #     for j, feat2 in enumerate(feature_names):
    #         for model, color in zip(models, colors):
    #             model_name = sntypes_map[model]
    #             ax[i, j].scatter(XTrain[:, j][yTrain == model], XTrain[:, i][yTrain == model], color=color, marker='.', alpha=0.3, label="Train_%s" % model_name)
    #             ax[i, j].scatter(XTest[:, j][yTest == model], XTest[:, i][yTest == model], color=color, marker='.', label="Test_%s" % model_name)
    #             if i == len(feature_names) - 1:
    #                 ax[i, j].set_xlabel(feat2)
    #             if j == 0:
    #                 ax[i, j].set_ylabel(feat1)
    # ax[0, j].legend(loc='upper left', bbox_to_anchor=(1,1))
    # fig.savefig(os.path.join(fig_dir, 'feature_space'))

    c = ChainConsumer()
    for model in models[::-1]:
        model_name = sntypes_map[model]
        c.add_chain(XTrain[yTrain == model], parameters=feature_names, name=model_name)
    c.configure()
    fig = c.plotter.plot()
    fig.savefig(filename=os.path.join(fig_dir, 'feature_space_contours'), transparent=False)


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'classify')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_test.hdf5')
    sntypes_map = helpers.get_sntypes()

    data_release = '20180407'
    field = 'DDF'
    model = '%'
    #
    # feature_names = ('skew', 'kurtosis', 'stetsonk', 'shapiro', 'acorr', 'hlratio',
    #                  'rms', 'mad', 'amplitude', 'q31', 'entropy', 'von-neumann')
    passbands = ('r', 'i', 'z', 'Y')

    feature_names = sum([['variance_%s' % p, 'kurtosis_%s' % p, 'filt-variance_%s' % p, 'filt-kurtosis_%s' % p,
                           'shapiro_%s' % p, 'p-value_%s' % p, 'skew_%s' % p, 'q31_%s' % p,
                           'stetsonk_%s' % p, 'acorr_%s' % p, 'von-neumann_%s' % p, 'hlratio_%s' % p,
                           'amplitude_%s' % p, 'filt-amplitude_%s' % p,  'somean_%s' % p, 'rms_%s' % p, 'mad_%s' % p,
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

    feature_names = np.array(feature_names)

    X, y = get_labels_and_features(fpath, data_release, field, model, feature_names, passbands)

    # models = [3, 4, 41, 42, 45, 60, 61, 62, 63]
    models = [1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 80, 81, 90]
    classify(X, y, models, sntypes_map, feature_names, fig_dir)

    plt.show()


if __name__ == '__main__':
    main()
