import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import helpers

ROOT_DIR = '../..'  # os.getenv('PLASTICC_DIR')


def get_features(fpath, data_release, field='%', model='%', base='%'):
    """ Get features from hdf5 files. """
    hdffile = h5py.File(fpath, 'r')
    data = np.array(hdffile[data_release])
    hdffile.close()

    # Ignore points that have redshift == 0
    data = data[np.where(data['redshift'] != 0)[0]]

    return data


def plot_features(fpath, data_release, feature_names=('redshift',), field='DDF', model='1', fig_dir='.',
                  sntypes_map=None):
    model_name = sntypes_map[int(model)]
    features = get_features(fpath, data_release)

    features_dict = {'u': {}, 'g': {}, 'r': {}, 'i': {}, 'z': {}, 'Y': {}}
    for pb in features_dict.keys():
        for f in feature_names:
            if f in ['objid', 'redshift']:
                feat_name = "%s" % (f)
            else:
                feat_name = "%s_%s" % (f, pb)
            features_dict[pb][f] = features[feat_name]

    with PdfPages(f'{fig_dir}/{model_name}_{data_release}_{field}.pdf') as pdf:
        for pb in features_dict.keys():
            fig, ax = plt.subplots(len(feature_names) - 1, sharex=True, figsize=(8, 15))
            for i, f in enumerate(feature_names[1:]):
                if f != 'redshift':
                    ax[i].scatter(features_dict[pb]['redshift'], features_dict[pb][f], marker='.', alpha=0.1)
                    ax[i].set_ylabel(f, rotation=0, labelpad=30)
            ax[-1].set_xlabel('redshift')
            ax[0].set_title("{} {}".format(model_name, pb))
            fig.subplots_adjust(hspace=0)
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
            fig.savefig("{0}/features_{1}_{2}_{3}.png".format(fig_dir, field, model_name, pb), bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures', 'features')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fpath = os.path.join(ROOT_DIR, 'plasticc', 'features_all_DDF.hdf5')
    sntypes_map = helpers.get_sntypes()

    feature_names = ('redshift', 'variance', 'kurtosis', 'amplitude', 'skew', 'somean', 'shapiro', 'q31',
                     'rms', 'mad', 'stetsonj', 'stetsonk', 'acorr', 'hlratio')

    for data_release in ['20180221']:
        for field in ['DDF']:
            for model in [1]:
                plot_features(fpath, data_release, feature_names, field, model, fig_dir, sntypes_map)

    plt.show()


if __name__ == '__main__':
    main()
