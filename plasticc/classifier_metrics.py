import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from chainconsumer import ChainConsumer


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdBu, fig_dir='.'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # Multiply off diagonal by -1
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1
    print(cm)

    fig = plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(abs(cm[i, j]), fmt), horizontalalignment="center",
                 color="white" if abs(cm[i, j]) > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix'))


def plot_feature_importance(classifier, feature_names, num_features, fig_dir):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_features):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    fig = plt.figure(figsize=(20, 10))
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices], color='#2ca02c', yerr=std[indices], align="center")
    plt.xticks(range(num_features), feature_names[indices], rotation=90)
    plt.tick_params(axis='both', labelsize=11)
    plt.xlim([-1, num_features])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance'))


def plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir, add_save_name=''):
    c = ChainConsumer()
    for model in models[::-1]:
        model_name = sntypes_map[model]
        c.add_chain(X[y == model], parameters=list(feature_names), name=model_name)
    c.configure()
    fig = c.plotter.plot()
    fig.savefig(fname=os.path.join(fig_dir, 'feature_space_contours_%s' % add_save_name), transparent=False)
