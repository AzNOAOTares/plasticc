import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn import over_sampling

from plot_features import get_features
import helpers
from classifier_metrics import plot_features_space, plot_feature_importance, plot_confusion_matrix
import seaborn as sns

sns.reset_orig()

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
    num_features = X.shape[1]

    # Remove models before training:
    remove_models = []
    # Remove models with fewer than 5 objects (as SMOTE can't work with that few objects)
    for m in models:
        nobs = len(X[y == m])
        print(m, nobs)
        if nobs <= 5:
            print("Removing model {}, because it only has {} objects.".format(m, nobs))
            remove_models.append(m)
    for m in remove_models:
        mask = np.where(y != m)[0]
        X = X[mask]
        y = y[mask]
        models.remove(m)
    model_names = [sntypes_map[model] for model in models]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, shuffle=True)

    # SMOTE to correct for imbalanced data on training set only
    sm = over_sampling.SMOTE(random_state=42)
    X_train, y_train = sm.fit_sample(X_train, y_train)

    # Train model
    ml_model = RandomForestClassifier(n_estimators=200)
    ml_model.fit(X_train, y_train)

    # Get Accuracy
    y_pred = ml_model.predict(X_test)
    accuracy = len(np.where(y_pred == y_test)[0])
    print("Accuracy is: {}/{} = {}".format(accuracy, len(y_pred), accuracy / len(y_pred)))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # visualize test performance
    plot_feature_importance(ml_model, feature_names, num_features, fig_dir)
    plot_confusion_matrix(cnf_matrix, classes=model_names, normalize=True, title='Normalized confusion matrix', fig_dir=fig_dir)
    plot_features_space(models, sntypes_map, X_train, y_train, feature_names, fig_dir)

    return y_pred


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'classify')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_test.hdf5')
    sntypes_map = helpers.get_sntypes()

    data_release = '20180407'
    field = 'DDF'
    model = '%'

    passbands = ('r', 'i', 'z', 'Y')

    feature_names = sum([['variance_%s' % p, 'kurtosis_%s' % p, 'filt-kurtosis_%s' % p,
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

    feature_names = np.array(feature_names)

    X, y = get_labels_and_features(fpath, data_release, field, model, feature_names, passbands)

    models = [1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 64, 80, 81, 90]
    classify(X, y, models, sntypes_map, feature_names, fig_dir)

    plt.show()


if __name__ == '__main__':
    main()
