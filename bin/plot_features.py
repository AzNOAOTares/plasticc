import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import helpers

ROOT_DIR = '..' # os.getenv('PLASTICC_DIR')


def get_features(fpath, data_release, field_in='%', model_in='%'):
    """ Get features from hdf5 files. """
    hdffile = h5py.File(fpath, 'r')
    features = np.array(hdffile[data_release])
    hdffile.close()

    indexes = []
    for i, objid in enumerate(features['objid']):
        field, model, base, snid = objid.astype(str).split('_')
        if (field == field_in or field_in == '%') and (model_in == '%' or int(model) == int(model_in)):
            indexes.append(i)

    features = features[indexes]

    return features


def plot_features(fpath, data_release, feature_names=('redshift',), field='DDF', model='1', fig_dir='.',
                  sntypes_map=None):
    model_name = sntypes_map[int(model)]
    features = get_features(fpath, data_release, field, model)
    passbands = ('u', 'g', 'r', 'i', 'z')

    features_dict = {'u': {}, 'g': {}, 'r': {}, 'i': {}, 'z': {}, 'Y': {}}
    for pb in features_dict.keys():
        for f in feature_names:
            if f in ['objid', 'redshift']:
                feat_name = "%s" % f
            else:
                feat_name = "%s_%s" % (f, pb)
            features_dict[pb][f] = features[feat_name]

    xlabel = 'redshift'
    with PdfPages(f'{fig_dir}/{model_name}_{data_release}_{field}.pdf') as pdf:
        for pb in passbands:
            fig, ax = plt.subplots(len(feature_names) - 2, sharex=True, figsize=(8, 15))
            for i, f in enumerate(feature_names[2:]):
                if f != xlabel:
                    x = np.copy(features_dict[pb][xlabel])
                    y = np.copy(features_dict[pb][f])
                    mask = ~np.isnan(y)  # Mask NaN points
                    x, y = x[mask], y[mask]
                    objid_masked = np.copy(features_dict[pb]['objid'])[mask]
                    data = np.array([x, y])
                    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
                    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    kernel = stats.gaussian_kde(data)
                    Z = np.reshape(kernel(positions).T, X.shape)
                    cfset = ax[i].contourf(X, Y, Z, cmap='Blues')
                    # cset = ax[i].contour(X, Y, Z, colors='k')

                    # ax[i].scatter(x, y, marker='.', alpha=0.2)
                    ax[i].set_ylabel(f, rotation=0, labelpad=30)

                    # Find plotting range by removing points over 10 standard deviations from the median 3 times iteratively.
                    for ii in range(1):
                        ystd = np.nanstd(y)
                        ymedian = np.nanmedian(y)
                        for jj in np.where(abs(y - ymedian) > 10*ystd)[0]:
                            print("Extreme values for: ", objid_masked[jj], pb, f, y[jj])
                        mask = np.where(abs(y - ymedian) < 10*ystd)[0]
                        y = y[mask]
                    ax[i].set_ylim([y.min(), y.max()])

            ax[-1].set_xlabel(xlabel)
            ax[0].set_title("{} {}".format(model_name, pb))
            fig.subplots_adjust(hspace=0)
            # plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
            fig.savefig("{0}/features_{1}_{2}_{3}.png".format(fig_dir, field, model_name, pb), bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'features4')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_all_DDF.hdf5')
    sntypes_map = helpers.get_sntypes()

    feature_names = ('objid', 'redshift', 'variance', 'kurtosis', 'amplitude', 'skew', 'somean', 'shapiro', 'q31',
                     'rms', 'mad', 'stetsonj', 'stetsonk', 'acorr', 'hlratio')

    for data_release in ['20180221']:
        for field in ['DDF']:
            for model in [1, 2, 3, 42, 45, 60, 61, 62, 63]:
                plot_features(fpath, data_release, feature_names, field, model, fig_dir, sntypes_map)

    plt.show()


if __name__ == '__main__':
    main()
