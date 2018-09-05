import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from chainconsumer import ChainConsumer


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdBu, fig_dir='.', name='', combine_kfolds=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    np.savetxt(os.path.join(fig_dir, 'confusion_matrix_raw_%s.csv' % name), cm)

    if combine_kfolds:
        uncertainties = np.std(cm, axis=0)
        cm = np.sum(cm, axis=0)

    if normalize:
        if combine_kfolds:
            uncertainties = uncertainties.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # Multiply off diagonal by -1
    off_diag = ~np.eye(cm.shape[0], dtype=bool)
    cm[off_diag] *= -1
    np.savetxt(os.path.join(fig_dir, 'confusion_matrix_%s.csv' % name), cm)
    print(cm)

    fig = plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=-1, vmax=1)
    # plt.title(title)
    cb = plt.colorbar()
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = format(abs(cm[i, j]), fmt)
        if combine_kfolds:
            unc = format(uncertainties[i, j], fmt)
            cell_text = r"{} $\pm$ {}".format(value, unc)
        else:
            cell_text = value
        plt.text(j, i, cell_text, horizontalalignment="center",
                 color="white" if abs(cm[i, j]) > thresh else "black", fontsize=18)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix_%s.pdf' % name))


def plot_feature_importance(classifier, feature_names, num_features, fig_dir, num_features_plot=50, name=''):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(num_features):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    n = num_features_plot
    # Plot the feature importances of the forest
    fig = plt.figure(figsize=(20, 10))
    plt.title("Feature importances")
    plt.bar(range(n), importances[indices][:n], color='#2ca02c', yerr=std[indices][:n], align="center")
    plt.xticks(range(n), feature_names[indices][:n], rotation=90)
    plt.tick_params(axis='both', labelsize=13)
    plt.xlim([-1, n])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importance{}.pdf'.format(name)))


def scatter(x, colors, feature_names, models, sntypes_map):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    import seaborn as sns
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})

    xaxisclass = 0
    yaxisclass = 2

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,xaxisclass], x[:,yaxisclass], lw=0, s=10,
                    c=palette[colors.astype(np.int)], alpha=0.4)
    plt.xlim(min(x[:,xaxisclass]), max(x[:,yaxisclass]))
    plt.ylim(min(x[:,xaxisclass]), max(x[:,yaxisclass]))
    # ax.axis('off')
    ax.axis('tight')
    plt.xlabel(feature_names[xaxisclass])
    plt.ylabel(feature_names[yaxisclass])

    # We add the labels for each digit.
    txts = []
    for i, model in enumerate(models):
        model_name = sntypes_map[model]
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :][:,[xaxisclass,yaxisclass]], axis=0)
        txt = ax.text(xtext, ytext, model_name, fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def plot_features_space(models, sntypes_map, X, y, feature_names, fig_dir, add_save_name=''):
    colors = np.copy(y)
    for i, m in enumerate(models):
        colors[y == m] = i

    scatter(X, colors, feature_names, models, sntypes_map)
    plt.savefig(fname=os.path.join(fig_dir, 'feature_space_sns_%s' % add_save_name), dpi=120)
    plt.show()

    print(X.shape, y.shape)
    for i in range(len(X[0])):
        mask = np.where(X[:,i] != 0)[0]
        X = X[mask]
        y = y[mask]
    print(X.shape, y.shape)

    for i, f in enumerate(feature_names):
        feature_names[i] = "$" + f + "$"

    c = ChainConsumer()
    for model in models[::-1]:
        model_name = sntypes_map[model]
        c.add_chain(X[y == model], parameters=list(feature_names), name=model_name)
    c.configure(colors=["#B32222", "#D1D10D", "#455A64", "#1f77b4", "#2ca02c"], shade=True, shade_alpha=0.2, bar_shade=True)
    fig = c.plotter.plot()
    fig.savefig(fname=os.path.join(fig_dir, 'feature_space_contours_%s.pdf' % add_save_name), transparent=False)
