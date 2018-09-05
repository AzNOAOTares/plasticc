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
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, RepeatedKFold
from sklearn.externals import joblib

from plasticc.read_features import get_features, get_feature_names
from plasticc import helpers
from plasticc.classifier_metrics import plot_feature_importance, plot_confusion_matrix, plot_features_space
# from .pca_components import get_pca_features
import seaborn as sns
import pandas as pd

sns.reset_orig()

ROOT_DIR = '..'# os.getenv('PLASTICC_DIR')


def get_labels_and_features(fpath, data_release, field, model, feature_names, aggregate_classes=False, pca=False, helpers=None):
    X = []  # features
    y = []  # labels
    agg_map = helpers.aggregate_sntypes()

    features = get_features(fpath, data_release, field, model, aggregate_classes=False, helpers=helpers)

    # # Remove features that contain more have more than 5000 objects with NaNs
    # for name, nan_count in pd.DataFrame(features).isnull().sum().iteritems():
    #     if nan_count > 5000:
    #         print("Removing feature", name, "because it has", nan_count, "nan value objects", "out of", len(features))
    #         features = helpers.remove_field_name(features, name)
    #         feature_names = np.delete(feature_names, np.argwhere(feature_names == name))

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

    X = np.nan_to_num(X)

    print("Num objects before removing objects where any features are NaN: ", len(X))
    # Remove rows that contain any NaNs in them
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    print("Num objects after removing objects where any features are NaN: ", len(X))

    # Check for infinities

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
            mask = np.where(abs(X[:, f] - median) <= 20 * std)[0]
            if len(mask) < len(X) or len(mask) == 0:
                pass
            if np.where(abs(X[:, f] - median) > 20 * std)[0].any():
                pass
            X = X[mask]
            y = y[mask]

    return X, y, feature_names


def remove_redundant_classes(X, y, models, remove_models):
    # Remove models with fewer than 5 objects (as SMOTE can't work with that few objects)
    for m in models:
        nobs = len(X[y == m])
        print(m, nobs)
        if nobs <= 12:
            print("Removing model {}, because it only has {} objects.".format(m, nobs))
            remove_models.append(m)
    # Remove models in X,y that are not in models
    for m in set(y):
        if m not in models:
            print("Not including model {} in classifier, even though it's in the features table".format(m))
            remove_models.append(m)
    for m in remove_models:
        print(m, remove_models)
        mask = np.where(y != m)[0]
        X = X[mask]
        y = y[mask]
        if m in models:
            models.remove(m)

    return X, y, models, remove_models


def make_class_labels(models, model_names, X_train, y_train, X_test, y_test):
    """ Count number in each model. """
    nobs_train, nobs_test, model_labels = [], [], []
    for m in models:
        nobs_train.append(len(X_train[y_train == m]))
        nobs_test.append(len(X_test[y_test == m]))
    for i, m in enumerate(models):
        model_labels.append("{}\ntrain: {}\ntest: {}".format(model_names[i], nobs_train[i], nobs_test[i]))

    return model_labels, nobs_train, nobs_test


def oversampling(models, X_train, y_train):
    """SMOTE to correct for imbalanced data on training set only."""
    print("SMOTE...")
    sm = over_sampling.SMOTE(random_state=42, n_jobs=2)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    for m in models:
        nobs = len(X_train[y_train == m])
        print(m, nobs)

    return X_train, y_train


def save_truth_tables(classifier, X_test, y_test, name, models, fig_dir, numfold):
    pred_proba = classifier.predict_proba(X_test)
    truth_table = np.zeros(pred_proba.shape)
    for i, m in enumerate(y_test):
        truth_table[i][models.index(m)] = 1
    np.savetxt(os.path.join(fig_dir, 'predicted_prob_{}_kfold_{}.csv'.format(name, numfold)), pred_proba)
    np.savetxt(os.path.join(fig_dir, 'truth_table_{}_kfold_{}.csv'.format(name, numfold)), truth_table)


def save_tree_diagram(classifier, feature_names, out_file='tree', fig_dir='.'):
    dotfile = os.path.join(fig_dir, "{}.dot".format(out_file))
    imagefile = os.path.join(fig_dir, "{}.pdf".format(out_file))

    from sklearn.tree import export_graphviz
    export_graphviz(classifier.estimators_[3],
                    out_file=dotfile,
                    feature_names=feature_names,
                    filled=True,
                    rounded=True)
    # import pydot
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('tree.png')
    os.system('dot -Tpdf {0} -o {1}'.format(dotfile, imagefile))


def remove_objects_with_val(X, y, val):
    print(X.shape, y.shape)
    for i in range(len(X[0])):
        mask = np.where(X[:,i] != val)[0]
        X = X[mask]
        y = y[mask]
    print(X.shape, y.shape)

    return X, y


def get_n_best_features(n, X, y, classifier, feature_names, num_features, fig_dir, name, models, model_names):
    # Classify on smaller feature list
    c = classifier
    classifier = RandomForestClassifier(n_estimators=c.n_estimators, n_jobs=c.n_jobs, random_state=c.random_state, max_leaf_nodes=c.max_leaf_nodes, max_depth=c.max_depth, min_samples_split=c.min_samples_split, min_samples_leaf=c.min_samples_leaf, min_weight_fraction_leaf=c.min_weight_fraction_leaf, max_features=c.max_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, shuffle=True, random_state=42)
    sm = over_sampling.SMOTE(random_state=42, n_jobs=2)
    X_train, y_train = sm.fit_sample(X_train, y_train)
    model_labels, nobs_train, nobs_test = make_class_labels(models, model_names, X_train, y_train, X_test, y_test)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = len(np.where(y_pred == y_test)[0])
    print("Accuracy is: {}/{} = {}".format(accuracy, len(y_pred), accuracy / len(y_pred)))
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_feature_importance(classifier, feature_names, num_features, fig_dir)
    plot_confusion_matrix(cnf_matrix, classes=model_labels, normalize=True, title='Normalized confusion matrix_%s' % name, fig_dir=fig_dir, name=name)

    # Use top n features
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(indices)
    feature_names = feature_names[indices][:n]
    print(feature_names)
    X = X[:, indices][:, :n]
    num_features = n

    return num_features, feature_names, X


def hyper_parameter_tuning(X, y, classifier, models, sntypes_map, feature_names, fig_dir='.', remove_models=(), name=''):
    """
    Use a random grid of hyperparameters to search for best hyperparameters.

    Good example on RandomizedSearchCV here:
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """

    # Hyperparameter grid
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Get data
    num_features = X.shape[1]
    model_names = [sntypes_map[model] for model in models]
    X, y, models, remove_models = remove_redundant_classes(X, y, models, remove_models)

    # Get best features
    n = 50
    num_features, feature_names, X = get_n_best_features(n, X, y, classifier, feature_names, num_features, fig_dir, name, models, model_names)

    # Randomised Search
    clf_random = RandomizedSearchCV(estimator=classifier, param_distributions=random_grid, n_iter=7, cv=3, verbose=2,
                                   random_state=42, n_jobs=2)
    clf_random.fit(X, y)
    print(clf_random.best_params_)

    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy

    best_random = clf_random.best_estimator_
    # random_accuracy = evaluate(best_random, test_features, test_labels)


def classify(X, y, classifier, models, sntypes_map, feature_names, fig_dir='.', remove_models=(), name='', k=1):
    num_features = X.shape[1]

    X, y, models, remove_models = remove_redundant_classes(X, y, models, remove_models)

    model_names = [sntypes_map[model] for model in models]

    # Plot feature space before oversampling
    # plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir)

    # Get best features
    n = 50
    num_features, feature_names, X = get_n_best_features(n, X, y, classifier, feature_names, num_features, fig_dir, name, models, model_names)

    # Store each k fold info:
    kfold_classifiers = []
    kfold_cnf_matrices = []

    # Split into train/test or k-fold
    if k == 1:
        data = train_test_split(X, y, train_size=0.60, shuffle=True, random_state=42)
    else:
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        data = kfold.split(X=X, y=y)

    for numfold, datum in enumerate(data):
        if k == 1:
            X_train, X_test, y_train, y_test = data
        else:
            train_idx, test_idx = datum
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            print("Fold number:", numfold)
            c = classifier
            classifier = RandomForestClassifier(n_estimators=c.n_estimators, n_jobs=c.n_jobs, random_state=c.random_state+numfold, max_leaf_nodes=c.max_leaf_nodes, max_depth=c.max_depth, min_samples_split=c.min_samples_split, min_samples_leaf=c.min_samples_leaf, min_weight_fraction_leaf=c.min_weight_fraction_leaf, max_features=c.max_features)

        # Class labels and count train/test objects
        model_labels, nobs_train, nobs_test = make_class_labels(models, model_names, X_train, y_train, X_test, y_test)

        # SMOTE
        X_train, y_train = oversampling(models, X_train, y_train)

        if classifier == 'voting':
            clf1 = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
            clf2 = MLPClassifier()
            clf3 = GaussianNB()
            clf4 = KNeighborsClassifier(3, n_jobs=-1)
            classifier = VotingClassifier(estimators=[('RF', clf1), ('MLP', clf2), ('KNN', clf4)], voting='soft')

        # Train model
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)

        # Get Accuracy
        y_pred = classifier.predict(X_test)
        accuracy = len(np.where(y_pred == y_test)[0])
        print("Accuracy with top features is: {}/{} = {}".format(accuracy, len(y_pred), accuracy / len(y_pred)))

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        print(cnf_matrix)

        # Save probability arrays
        # save_truth_tables(classifier, X_test, y_test, name, models, fig_dir)

        # Store kfold metrics
        kfold_cnf_matrices.append(cnf_matrix)
        kfold_classifiers.append(classifier)

        if k == 1:
            break

    # Save model
    joblib.dump(classifier, os.path.join(fig_dir, 'classifier.joblib'))

    # visualize test performance
    if k == 1:
        combine_kfolds = False
    else:
        combine_kfolds = True
        cnf_matrix = kfold_cnf_matrices
    if hasattr(classifier, "feature_importances_"):
        plot_feature_importance(classifier, feature_names, num_features, fig_dir, num_features_plot=n, name=name + '_with_top_{}features'.format(n))
    plot_confusion_matrix(cnf_matrix, classes=model_labels, normalize=True, title='Normalized confusion matrix_{}_with_top_{}'.format(name, n), fig_dir=fig_dir, name=name + '_with_top_{}features'.format(n), combine_kfolds=combine_kfolds)

    # Save tree diagram
    save_tree_diagram(classifier, feature_names, out_file='tree_{}'.format(name), fig_dir=fig_dir)

    # Plot feature space
    # X, y = remove_objects_with_val(X, y, val=-99)
    # plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir, add_save_name='')

    return classifier, X_train, y_train, X_test, y_test, score, y_pred


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'classify_20180727_DDF')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_DDF_20180727.hdf5')
    sntypes_map = helpers.get_sntypes()

    data_release = '20180727'
    field = 'DDF'
    model = '%'
    passbands = ('u', 'g', 'r', 'i', 'z', 'Y')
    # models = [1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 64, 80, 81, 90, 91]
    # models = [1, 3, 6, 41, 45, 51, 60, 61, 62, 63, 64, 80, 81, 90, 91, 102, 103]
    models = [1, 5, 6, 41, 43, 45, 50, 51, 60, 64, 70, 80, 81, 83, 84, 91, 93, 99]
    remove_models = []
    feature_names = get_feature_names(passbands, ignore=('objid',))
    X, y, feature_names = get_labels_and_features(fpath, data_release, field, model, feature_names, aggregate_classes=True, pca=False, helpers=helpers)

    classifiers = [('RandomForest', RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)),
                   ('KNeighbors_10', KNeighborsClassifier(10)),
                   ('MLPNeuralNet', MLPClassifier()),
                   ('NaiveBayes', GaussianNB()),
                   ('QDA', QuadraticDiscriminantAnalysis()),
                   ('GaussianProcesses', GaussianProcessClassifier(1.0 * RBF(1.0))),
                   ('AdaBoost', AdaBoostClassifier()),
                   ('Linear SVM', SVC(kernel="linear", C=0.025)),
                   ('RBF SVM', SVC(gamma=2, C=1)),
                   ('DecisionTree', DecisionTreeClassifier(max_depth=5)),
                   ('voting', 'voting')]

    for name, classifier in classifiers[0:1]:
        print(name)
        fig_dir_classifier = os.path.join(fig_dir, name)
        if not os.path.exists(fig_dir_classifier):
            os.makedirs(fig_dir_classifier)
        classify(X, y, classifier, models, sntypes_map, feature_names, fig_dir_classifier, remove_models, name)


if __name__ == '__main__':
    main()
    plt.show()
