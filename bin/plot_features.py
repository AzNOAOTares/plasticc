import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels import robust
import pandas as pd
import seaborn as sns
import helpers

ROOT_DIR = '..'  # os.getenv('PLASTICC_DIR')


def get_features(fpath, data_release, field_in='%', model_in='%'):
    """ Get features from hdf5 files. """
    hdffile = h5py.File(fpath, 'r')
    features = np.array(hdffile[data_release])
    hdffile.close()

    indexes = []
    modelList = []
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        modelList.append(model)
        if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(model_in)):
            indexes.append(i)

    features = features[indexes]

    # rescaled_flux_new = []
    # rescaled_flux = features['rescaled-flux_r']
    # for s in rescaled_flux:
    #     s = s.decode('utf-8').replace('[', '')
    #     s = s.replace(']', '')
    #     s = s.replace('\\n', '')
    #     s = np.array(s.split()).astype(float)
    #     # rescaled_flux_new.append(s)
    #     rescaled_flux_new += list(s)
    # rescaled_flux_new = np.asarray(rescaled_flux_new)
    #
    # fig = plt.figure()
    #
    # num_bins = 50
    # # the histogram of the data
    # # n, bins, patches = plt.hist(rescaled_flux_new, num_bins, normed=1, facecolor='green', alpha=0.5)
    # sns.kdeplot(rescaled_flux_new)
    # plt.xlabel('rescaled-flux')
    # plt.title(model_in)
    # plt.show()

    return features

def convert_rescaled_flux_to_array(rescaled_flux_str_array):
    rescaled_flux_new = []
    for s in rescaled_flux_str_array:
        s = s.decode('utf-8').replace('[', '')
        s = s.replace(']', '')
        s = s.replace('\\n', '')
        try:
            s = np.array(s.split()).astype(float)
        except Exception as err: # Weird value error due to trailing '-' in array
            print(s, err)
            continue
        # rescaled_flux_new.append(s)
        rescaled_flux_new += list(s)
    rescaled_flux_new = np.asarray(rescaled_flux_new)
    return rescaled_flux_new

def get_features_dict(fpath, data_release, feature_names=('redshift',), field='DDF', model='1'):

    features = get_features(fpath, data_release, field, model)

    features_dict = {'u': {}, 'g': {}, 'r': {}, 'i': {}, 'z': {}, 'Y': {}}
    for pb in features_dict.keys():
        for f in feature_names:
            if f in ['objid', 'redshift']:
                feat_name = "%s" % f
            else:
                feat_name = "%s_%s" % (f, pb)
            features_dict[pb][f] = features[feat_name]

    return features_dict


def plot_features(fpath, data_release, feature_names=('redshift',), field='DDF', model='1', fig_dir='.', sntypes_map=None):

    model_name = sntypes_map[int(model)]
    passbands = ('u', 'g', 'r', 'i', 'z', 'Y')
    features_dict = get_features_dict(fpath, data_release, feature_names, field, model)

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
        minmax = {'kurtosis': (None, 6), 'amplitude': (None, None), 'skew': (None, None), 'somean': (-1, 2),
                  'shapiro': (None, None), 'q31': (None, 1), 'rms': (None, None), 'mad': (None, None), 'stetsonj': (0, 400),
                  'stetsonk': (None, None), 'acorr': (None, 6), 'hlratio': (None, 6), 'entropy': (None, 12),
                  'von-neumann': (None, None), 'variance': (None, None), 'rescaled-flux': (-2, 2), 'nobs4096': (None, None)}

        ymin, ymax = minmax[feature]
    # ymin, ymax = np.percentile(y, 0), np.percentile(y, 99)

    return ymin, ymax


def plot_features_joy_plot(fpath, data_release, feature_names=('redshift',), field='DDF', fig_dir='.', sntypes_map=None):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    passbands = ('u', 'g', 'r', 'i', 'z', 'Y')
    model_names = []
    features_by_model = {}
    for model in [1, 2, 3, 4, 5, 41, 42, 45, 60, 61, 62, 63]: #[1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 80, 81, 82, 90, 91]:
        model_name = sntypes_map[int(model)]
        model_names.append(model_name)
        features_by_pb = get_features_dict(fpath, data_release, feature_names, field, model)
        features_by_model[model_name] = features_by_pb

    features_by_model = pd.DataFrame(features_by_model)  # DF structure eg: [SNIbc: Y: objid]
    features_by_model = features_by_model.transpose()  # DF structure eg: [Y: SNIbc: objid]

    # Convert to 3D DataFrame instead of 2D dataframe of dicts
    for pb in passbands:
        for model_name in model_names:
            features_by_model[pb][model_name] = pd.DataFrame(features_by_model[pb][model_name])

    # Plotting joyplots for each feature and pb
    for pb in passbands:
        with PdfPages(f'{fig_dir}/{pb}_{field}_{data_release}.pdf') as pdf:
            for feature in feature_names[2:]:
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
                g.fig.suptitle("{} {}".format(feature, pb))
                # g.fig.savefig("{0}/{1} {2}".format(fig_dir, feature, pb))
                pdf.savefig(g.fig)


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'features_DDF')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_DDF.hdf5')
    sntypes_map = helpers.get_sntypes()

    # feature_names = ('objid', 'redshift', 'stetsonj')
    feature_names = ('objid', 'redshift', 'skew', 'kurtosis', 'stetsonk', 'shapiro', 'acorr', 'hlratio',
                     'rms', 'mad', 'somean', 'amplitude', 'q31', 'entropy', 'von-neumann', 'nobs4096')

    # for data_release in ['20180316']:
    #     for field in ['DDF']:
    #         for model in [1, 2, 3, 4, 5, 41, 42, 45, 50, 60, 61, 62, 63, 80, 81, 82, 90, 91]:
    #             plot_features(fpath, data_release, feature_names, field, model, fig_dir, sntypes_map)

    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'features_DDF')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plot_features_joy_plot(fpath, '20180316', feature_names, 'DDF', fig_dir, sntypes_map)

    plt.show()


if __name__ == '__main__':
    main()
