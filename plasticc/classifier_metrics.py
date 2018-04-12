import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from chainconsumer import ChainConsumer


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, fig_dir='.'):
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

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix'))


def plot_feature_importance(model_ml, feature_names, num_features, fig_dir):
    importances = model_ml.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_ml.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]


    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_features):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure(figsize=(15, 8))
    plt.title("Feature importances")
    plt.bar(range(num_features), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(num_features), feature_names[indices], rotation='vertical')
    plt.xlim([-1,num_features])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance'))


def plot_features_space(models, sntypes_map, XTrain, yTrain, feature_names, fig_dir):
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