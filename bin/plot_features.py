import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
import warnings
import matplotlib.pyplot as plt
import numpy as np
from plasticc import database
from plasticc.get_data import GetData


def get_features(table_name='features', columns=None, field='%', model='%', base='%', snid='%', sntype='%',
                 passband='%', limit=None, shuffle=False, sort=True, offset=0):
    """ Gets the header data given specific conditions.

    Parameters
    ----------
    table_name : str
        The name of the table to get features from. E.g. table_name='features'
    columns : list
        A list of strings of the names of the columns you want to retrieve from the database.
        E.g. columns=['nobs', 'variance', 'amplitude'].
    field : str, optional
        The field name. E.g. field='DDF' or field='WFD'. The default is '%' indicating that all fields will be included.
    model : str, optional
        The model number. E.g. model='04'. The default is '%' indicating that all model numbers will be included.
    base : str, optional
        The base name. E.g. base='NONIa'. The default is '%' indicating that all base names will be included.
    snid : str, optional
        The transient id. E.g. snid='87287'. The default is '%' indicating that all snids will be included.
    sntype : str, optional
        The transient type/class. E.g. sntype='3'. The default is '%' indicating that all sntypes will be included.
    passband : str, optional
        The passband filter. E.g. passband='u'
    limit : int, optional
        Limit the results to this number (> 0)
    shuffle : bool, optional
        Randomize the order of the results - not allowed with `sort`
    sort : bool, optional
        Order the results by objid - overrides `shuffle` if both are set
    offset : int, optional
        Start returning MySQL results from this row number offset
    Return
    -------
    result: numpy array
        2D array containing the features as columns
    """
    if columns is None:
        columns = ['objid']

    try:
        limit = int(limit)
        if limit <= 0:
            raise RuntimeError('prat')
    except Exception as e:
        limit = None

    try:
        offset = int(offset)
        if offset <= 0:
            raise RuntimeError('prat')
    except Exception as e:
        offset = None

    if limit is not None and shuffle is False and sort is False:
        sort = True

    sntype_command = '' if sntype == '%' else " AND sntype={}".format(sntype)
    limit_command = '' if limit is None else " LIMIT {:d}".format(limit)
    offset_command = '' if offset is None else " OFFSET {:d}".format(offset)
    if model != '%':
        model = "{:02n}".format(int(model))

    if sort is True and shuffle is True:
        message = 'Cannot sort and shuffle at the same time! That makes no sense!'
        shuffle = False
        warnings.warn(message, RuntimeWarning)

    shuffle_command = '' if shuffle is False else " ORDER BY RAND()"
    sort_command = '' if sort is False else ' ORDER BY objid'
    extra_command = ''.join([sntype_command, sort_command, shuffle_command, limit_command, offset_command])

    query = "SELECT {} FROM {} WHERE objid LIKE '{}_{}_{}_{}' {};".format(', '.join(columns), table_name, field, model, base, snid, extra_command)
    feature_columns = database.exec_sql_query(query)

    num_lightcurves = len(feature_columns)

    if num_lightcurves > 0:
        return np.array(feature_columns)
    else:
        print("No light curves in the database satisfy the given arguments. "
              "field: {}, model: {}, base: {}, snid: {}, sntype: {}".format(field, model, base, snid, sntype))
        return []


def plot_features(table_name='features_20180221', feature_names=['redshift'], field='%', model='%', base='%', snid='%',
                  sntype='%', passband='%', limit=None, shuffle=False, sort=True, offset=0, fig_dir='.', sntypes_map=None):
    features = get_features(table_name=table_name, columns=feature_names, field=field, model=model, base=base,
                            snid=snid, sntype=sntype, passband=passband, limit=limit, shuffle=shuffle, sort=sort, offset=offset)

    sntype_name = sntypes_map[sntype]
    features_dict = {}
    for i, f in enumerate(feature_names):
        features_dict[f] = features[:, i]

    fig, ax = plt.subplots(len(feature_names)-1, sharex=True, figsize=(8, 15))
    for i, f in enumerate(feature_names[1:]):
        if f != 'redshift':
            ax[i].scatter(features_dict['redshift'], features_dict[f], marker='.', alpha=0.1)
            ax[i].set_ylabel(f, rotation=0, labelpad=30)
    ax[-1].set_xlabel('redshift')
    ax[0].set_title("{} {}".format(sntype_name, passband))
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.savefig("{0}/features_{1}_{2}.png".format(fig_dir, sntype_name, passband), bbox_inches='tight')


def main():
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    feature_names = ['redshift', 'variance', 'skewness', 'kurtosis', 'amplitude', 'skew', 'somean', 'shapiro', 'q31',
                  'rms', 'mad', 'stetsonj', 'stetsonk', 'acorr', 'hlratio']
    data_release = '20180221'
    table_name = 'features_{}'.format(data_release)

    getdata = GetData(data_release)
    sntypes_map = getdata.get_sntypes()

    for sntype in [1]:
        for pb in ['i', 'r', 'Y', 'u', 'g', 'z']:
            plot_features(table_name=table_name, feature_names=feature_names, field='DDF', sntype=sntype, passband=pb,
                          limit=None, fig_dir=fig_dir, sntypes_map=sntypes_map)

    plt.show()


if __name__ == '__main__':
    main()
