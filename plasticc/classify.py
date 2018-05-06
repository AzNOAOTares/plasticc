import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn import over_sampling

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from read_features import get_features, get_feature_names
import helpers
from classifier_metrics import plot_feature_importance, plot_confusion_matrix, plot_features_space
from pca_components import get_pca_features
import seaborn as sns
import pandas as pd

sns.reset_orig()

ROOT_DIR = '..'# os.getenv('PLASTICC_DIR')


def get_labels_and_features(fpath, data_release, field, model, feature_names, aggregate_classes=False, pca=False):
    X = []  # features
    y = []  # labels
    agg_map = helpers.aggregate_sntypes()

    features = get_features(fpath, data_release, field, model, aggregate_classes=False)

    # Remove features that contain more have more than 5000 objects with NaNs
    for name, nan_count in pd.DataFrame(features).isnull().sum().iteritems():
        if nan_count > 5000:
            print("Removing feature", name, "because it has", nan_count, "nan value objects", "out of", len(features))
            features = helpers.remove_field_name(features, name)
            feature_names = np.delete(feature_names, np.argwhere(feature_names == name))

    if pca:
        features = get_pca_features(features, n_comps=50, feature_names=feature_names)
        feature_names = np.array(features.dtype.names[1:])

    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        if aggregate_classes:
            model = agg_map[int(model)]
        if model == 'ignore':
            continue
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

    # Remove rows that contain are too big or too small
    mask = ~np.logical_or(X < -1e30, X > 1e30).any(axis=1)
    X = X[mask]
    y = y[mask]
    print("Num objects after removing objects where any features are greater than 1e99: ", len(X))

    # Remove extreme values over 20 standard deviations from the median 1 times iteratively
    for ii in range(1):
        for f in range(X.shape[1]):
            std = np.std(X[:, f])
            median = np.median(X[:, f])
            mask = np.where(abs(X[:, f] - median) < 20 * std)[0]
            if np.where(abs(X[:, f] - median) > 20 * std)[0].any():
                pass
            X = X[mask]
            y = y[mask]

    return X, y, feature_names


def classify(X, y, classifier, models, sntypes_map, feature_names, fig_dir='.', remove_models=()):
    num_features = X.shape[1]

    # Remove models with fewer than 5 objects (as SMOTE can't work with that few objects)
    for m in models:
        nobs = len(X[y == m])
        print(m, nobs)
        if nobs <= 9:
            print("Removing model {}, because it only has {} objects.".format(m, nobs))
            remove_models.append(m)
    for m in remove_models:
        mask = np.where(y != m)[0]
        X = X[mask]
        y = y[mask]
        models.remove(m)

    # # Plot feature space before oversampling
    # plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, shuffle=True)
    model_names = [sntypes_map[model] for model in models]

    # SMOTE to correct for imbalanced data on training set only
    sm = over_sampling.SMOTE(random_state=42, n_jobs=20)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    for m in models:
        nobs = len(X_train[y_train == m])
        print(m, nobs)

    # for n in [5, 10, 25, 50, 100, 500, 1000]:
    # classifier = KNeighborsClassifier(n)
    # clf1 = RandomForestClassifier(n_estimators=50, random_state=42)
    # for n in [3, 5, 10, 25, 50, 100, 500, 1000]:
    #     clf2 = KNeighborsClassifier(n_neighbors=n)
    #     classifier = VotingClassifier(estimators=[('RF', clf1), ('KNN', clf2)], voting='hard')

    # Train model
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    # Get Accuracy
    y_pred = classifier.predict(X_test)
    accuracy = len(np.where(y_pred == y_test)[0])
    print("Accuracy is: {}/{} = {}".format(accuracy, len(y_pred), accuracy / len(y_pred)))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # visualize test performance
    if hasattr(classifier, "feature_importances_"):
        plot_feature_importance(classifier, feature_names, num_features, fig_dir)
    plot_confusion_matrix(cnf_matrix, classes=model_names, normalize=True, title='Normalized confusion matrix', fig_dir=fig_dir)
    # plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir, add_save_name='')

    return classifier, X_train, y_train, X_test, y_test, score, y_pred


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'classify', 'ddf_with_cesium')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_ddf_with_cesium.hdf5')
    sntypes_map = helpers.get_sntypes()

    data_release = '20180407'
    field = 'DDF'
    model = '%'
    passbands = ('r', 'i', 'z', 'Y')
    # models = [1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 64, 80, 81, 90, 91]
    models = [1, 2, 41, 45, 50, 60, 61, 62, 63, 64, 80, 81, 90, 91]
    # models = [1, 2, 41, 45, 50, 60, 63, 64, 80, 81, 91, 200]
    remove_models = []
    feature_names = get_feature_names(passbands, ignore=('objid',))
    X, y, feature_names = get_labels_and_features(fpath, data_release, field, model, feature_names, aggregate_classes=True, pca=False)

    classifiers = [('RandomForest', RandomForestClassifier(n_estimators=50, n_jobs=-1)),
                   ('KNeighbors', KNeighborsClassifier(3)),
                   ('Linear SVM', SVC(kernel="linear", C=0.025)),
                   ('RBF SVM', SVC(gamma=2, C=1)),
                   ('GaussianProcesses', GaussianProcessClassifier(1.0 * RBF(1.0))),
                   ('DecisionTree', DecisionTreeClassifier(max_depth=5)),
                   ('NeuralNet', MLPClassifier(alpha=1)),
                   ('AdaBoost', AdaBoostClassifier()),
                   ('NaiveBayes', GaussianNB()),
                   ('QDA', QuadraticDiscriminantAnalysis())]

    for name, classifier in classifiers[0:1]:
        fig_dir_classifier = os.path.join(fig_dir, name)
        if not os.path.exists(fig_dir_classifier):
            os.makedirs(fig_dir_classifier)
        classify(X, y, classifier, models, sntypes_map, feature_names, fig_dir_classifier, remove_models)


if __name__ == '__main__':
    main()
    plt.show()
