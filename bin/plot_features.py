import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import pandas as pd
import seaborn as sns
from ..plasticc import helpers
from ..plasticc.read_features import get_features, get_feature_names

ROOT_DIR = '..'  # os.getenv('PLASTICC_DIR')


def convert_rescaled_flux_to_array(rescaled_flux_str_array):
    rescaled_flux_new = []
    for s in rescaled_flux_str_array:
        s = s.decode('utf-8').replace('[', '')
        s = s.replace(']', '')
        s = s.replace('\\n', '')
        try:
            s = np.array(s.split()).astype(float)
        except Exception as err:  # Weird value error due to trailing '-' in array
            print(s, err)
            continue
        # rescaled_flux_new.append(s)
        rescaled_flux_new += list(s)
    rescaled_flux_new = np.asarray(rescaled_flux_new)
    return rescaled_flux_new


def get_features_dict(fpath, data_release, feature_names=('redshift',), field='DDF', model='1', passbands=None, aggregate_classes=False):
    features = get_features(fpath, data_release, field, model, aggregate_classes=aggregate_classes, helpers=helpers)

    features_dict = {pb: {} for pb in passbands + ['general_features']}
    for feat in feature_names:
        if feat in ['objid']:
            continue
        elif feat[-2] == '_':
            pb = feat[-1]
            features_dict[pb][feat] = features[feat]
        elif feat[-2] == '-':
            features_dict['general_features'][feat] = features[feat]
        elif 'period' in feat or feat == 'redshift':
            features_dict['general_features'][feat] = features[feat]
        else:
            print("I've made a mistake, fix this...", feat)

    return features_dict


def plot_features(fpath, data_release, feature_names=('redshift',), field='DDF', model='1', fig_dir='.',
                  sntypes_map=None, passbands=('r',), aggregate_classes=False):
    model_name = sntypes_map[int(model)]
    features_dict = get_features_dict(fpath, data_release, feature_names, field, model, passbands, aggregate_classes)

    xlabel = 'redshift'
    with PdfPages(f'{fig_dir}/{model_name}_{data_release}_{field}.pdf') as pdf:
        for pb in passbands:
            fig, ax = plt.subplots(len(feature_names) - 2, sharex=True, figsize=(8, 15))
            for i, f in enumerate(feature_names[2:]):
                print(model, pb, f)
                if f not in features_dict[pb].keys():
                    continue
                if f != xlabel:
                    x = np.copy(features_dict[pb][xlabel])
                    y = np.copy(features_dict[pb][f])
                    if x.size == 0:
                        print("No entries for ", model_name, pb, f)
                        continue
                    mask = ~np.isnan(y)  # Mask NaN points
                    x, y = x[mask], y[mask]
                    objid_masked = np.copy(features_dict[pb]['objid'])[mask]
                    data = np.array([x, y])
                    try:
                        xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
                    except ValueError as err:
                        print("No entries: {}".format(err))
                        continue
                    if len(x) > 30:
                        X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                        positions = np.vstack([X.ravel(), Y.ravel()])
                        kernel = stats.gaussian_kde(data)
                        Z = np.reshape(kernel(positions).T, X.shape)
                        cfset = ax[i].contourf(X, Y, Z, cmap='Blues')
                        # cset = ax[i].contour(X, Y, Z, colors='k')
                    else:
                        ax[i].scatter(x, y, marker='.', alpha=0.2)
                    ax[i].set_ylabel(f, rotation=0, labelpad=30)

                    # # Find plotting range by removing points over 10 standard deviations from the median 3 times iteratively.
                    # for ii in range(1):
                    #     ystd = np.nanstd(y)
                    #     ymedian = np.nanmedian(y)
                    #     for jj in np.where(abs(y - ymedian) > 10*ystd)[0]:
                    #         print("Extreme values for: ", objid_masked[jj], pb, f, y[jj])
                    #     mask = np.where(abs(y - ymedian) <= 10*ystd)[0]
                    #     y = y[mask]
                    #     objid_masked = objid_masked[mask]
                    # ax[i].set_ylim([y.min(), y.max()])

            ax[-1].set_xlabel(xlabel)
            ax[0].set_title("{} {}".format(model_name, pb))
            fig.subplots_adjust(hspace=0)
            # plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
            # fig.savefig("{0}/features_{1}_{2}_{3}.png".format(fig_dir, field, model_name, pb), bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def get_limits(y, feature=None):
    # Find plotting range by removing points over 10 standard deviations from the median 3 times iteratively.
    # for ii in range(1):
    #     ymad = robust.mad(y)
    #     ymedian = np.nanmedian(y)
    #     mask = np.where(abs(y - ymedian) <= 10*ymad)[0]
    #     y = y[mask]
    # ymin, ymax = min(y), max(y)
    #
    if feature is not None:
        if 'amp ' in feature:
            feature = 'amp__'
        if feature in ['period1', 'period2', 'period3', 'period4', 'period5']:
            feature = 'period__'
        minmax = {'kurtosis': (None, 5), 'amplitude': (-10, 10), 'skew': (None, None), 'somean': (-1, 2),
                  'shapiro': (None, None), 'q31': (None, 1), 'rms': (None, None), 'mad': (None, None),
                  'stetsonj': (-10, 10),
                  'stetsonk': (None, None), 'acorr': (-6, 6), 'hlratio': (None, 6), 'entropy': (None, 12),
                  'von-neumann': (None, None), 'variance': (None, None), 'rescaled-flux': (-2, 2),
                  'nobs4096': (None, None),
                  'amp': (-8, 8), 'filt-kurtosis': (None, 5), 'filt-amplitude': (-10, 10), 'stetsonl': (-10, 10),
                  'period': (-1, 3)}
        try:
            ymin, ymax = minmax[feature[:-2]]
        except KeyError:
            # ymin, ymax = (None, None)
            datarange = np.percentile(y, 95) - np.percentile(y, 5)
            ymin, ymax = (np.percentile(y, 5) - datarange*3), (np.percentile(y, 95) + datarange*3)

    return ymin, ymax


def plot_features_joy_plot(fpath, data_release, feature_names=('redshift',), field='DDF', fig_dir='.', sntypes_map=None,
                           passbands=('r',), models=(1,), aggregate_classes=False):
    # sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    model_names = []
    features_by_model = {}
    for model in models:
        model_name = sntypes_map[int(model)]
        model_names.append(model_name)
        features_by_pb = get_features_dict(fpath, data_release, feature_names, field, model, passbands, aggregate_classes)
        features_by_model[model_name] = features_by_pb

    features_by_model = pd.DataFrame(features_by_model)  # DF structure eg: [SNIbc: Y: objid]
    features_by_model = features_by_model.transpose()  # DF structure eg: [Y: SNIbc: objid]

    # # Plot correlation matrix
    # plt.figure(figsize=(20, 20))
    # df = pd.DataFrame(features_by_model['r']['SN1a'])
    # # all_feat = features_by_model.transpose()['SN1a']
    # # df = pd.DataFrame({**all_feat['r'], **all_feat['i'], **all_feat['z'], **all_feat['Y'], **all_feat['general_features']})
    # import seaborn as sns
    # corr = df.corr()
    # sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values, vmin=-1, vmax=1)
    # sns.set(font_scale=1.9)
    # plt.yticks(rotation=0,fontsize=22)
    # plt.xticks(rotation=90,fontsize=22)
    # plt.tight_layout()
    # plt.savefig(os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'correlation', 'SN1a'))

    # Convert to 3D DataFrame instead of 2D dataframe of dicts
    for pb in passbands + ['general_features']:
        for model_name in model_names:
            features_by_model[pb][model_name] = pd.DataFrame(features_by_model[pb][model_name])

    # Plotting joyplots for each feature and pb
    for pb in passbands + ['general_features']:
        with PdfPages(f'{fig_dir}/{pb}_{field}_{data_release}.pdf') as pdf:
            for feature in feature_names[1:]:
                joyplot_data = {'g': [], 'x': []}
                for model_name in model_names:
                    if feature not in features_by_model[pb][model_name].keys():
                        continue
                    x_values = list(features_by_model[pb][model_name][feature].values)
                    if feature == 'rescaled-flux':
                        x_values = list(convert_rescaled_flux_to_array(x_values))
                    if len(x_values) == 0:
                        print("No entries for ", model_name, pb, feature)
                        continue
                    joyplot_data['x'] += x_values
                    joyplot_data['g'] += [model_name] * len(x_values)
                df = pd.DataFrame(joyplot_data)
                df = df.dropna(axis=0, how='any')  # Remove rows with NaN
                if df.empty:
                    continue
                nobs = {model_name: len(df['x'][df['g'] == model_name]) for model_name in model_names}
                print(pb, feature, nobs)

                # Using example joyplot: https://seaborn.pydata.org/examples/kde_joyplot.html
                # Initialize the FacetGrid object
                pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
                g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)

                # Draw the densities in a few steps
                g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
                g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
                g.map(plt.axhline, y=0, lw=2, clip_on=False)

                # Define and use a simple function to label the plot in axes coordinates
                def label(x, color, label):
                    ax = plt.gca()
                    ax.text(0, .2, label, fontweight="bold", color=color,
                            ha="left", va="center", transform=ax.transAxes)
                    ax.text(0.9, 0.2, "nobs: {}".format(nobs[label]), fontsize=9, transform=ax.transAxes)

                g.map(label, x='')

                # Set the subplots to overlap
                g.fig.subplots_adjust(hspace=-.25)

                # Remove axes details that don't play will with overlap
                g.set_titles("")
                g.set(yticks=[])
                g.despine(bottom=True, left=True)

                # Set limits
                xmin, xmax = get_limits(df['x'].values, feature)
                if feature in ['stetsonj']:
                    print('here')
                    g.set(xlim=(xmin, xmax), ylim=(0, 0.01))
                else:
                    g.set(xlim=(xmin, xmax))

                # Save figures
                g.fig.suptitle("{}".format(feature))
                # g.fig.savefig("{0}/{1} {2}".format(fig_dir, feature, pb))
                pdf.savefig(g.fig)


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'features_test')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_test.hdf5')
    sntypes_map = helpers.get_sntypes()

    passbands = ['u', 'g', 'r', 'i', 'z', 'Y']

    feature_names = get_feature_names(passbands, ignore=())

    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'features_test')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    models = [1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 80, 81, 90]
    # models = [1, 2, 41, 45, 50, 60, 61, 62, 63, 64, 80, 81, 90]
    plot_features_joy_plot(fpath, '20180407', feature_names, 'DDF', fig_dir, sntypes_map, passbands, models, aggregate_classes=False)


if __name__ == '__main__':
    main()
    sns.reset_orig()
    # plt.show()

