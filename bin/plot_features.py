import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
import warnings
import matplotlib.pyplot as plt
import numpy as np
from plasticc import database


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
    limit_command = '' if limit is None else " LIMIT {:n}".format(limit)
    offset_command = '' if offset is None else " OFFSET {:n}".format(offset)
    if model != '%':
        model = "{:02n}".format(int(model))

    if sort is True and shuffle is True:
        message = 'Cannot sort and shuffle at the same time! That makes no sense!'
        shuffle = False
        warnings.warn(message, RuntimeWarning)

    shuffle_command = '' if shuffle is False else " ORDER BY RAND()"
    sort_command  = '' if sort is False else ' ORDER BY objid'
    extra_command = ''.join([sntype_command, sort_command, shuffle_command, limit_command, offset_command])

    #query = "SELECT {0} FROM {1} WHERE objid LIKE '{2}%' AND objid LIKE '%{3}%' AND objid LIKE '%{4}%' AND objid LIKE '%{5}' {6};".format(', '.join(columns), self.data_release, field, model, base, snid, extra_command)
    query = "SELECT {} FROM {} WHERE objid LIKE '{}_{}_{}_{}_{}' {};".format(', '.join(columns), table_name, field, model, base, snid, passband, extra_command)
    feature_columns = database.exec_sql_query(query)

    num_lightcurves = len(feature_columns)

    if num_lightcurves > 0:
        return np.array(feature_columns)
    else:
        print("No light curves in the database satisfy the given arguments. "
              "field: {}, model: {}, base: {}, snid: {}, sntype: {}".format(field, model, base, snid, sntype))
        return []


def plot_features(table_name='features', feature_names=['redshift',], field='%', model='%', base='%', snid='%', sntype='%',
                  limit=None, shuffle=False, sort=True, offset=0, fig_dir='.'):
    features = get_features(table_name=table_name, columns=feature_names, field=field, model=model, base=base,
                            snid=snid, sntype=sntype, limit=limit, shuffle=shuffle, sort=sort, offset=offset)

    features_dict = {}
    for i, f in enumerate(feature_names):
        features_dict[f] = features[:, i]

    fig, ax = plt.subplots(len(feature_names)-1, sharex=True, figsize=(8, 15))
    for i, f in enumerate(feature_names[1:]):
        if f != 'redshift':
            ax[i].scatter(features_dict['redshift'], features_dict[f], marker='.', alpha=0.2)
            ax[i].set_ylabel(f, rotation=0, labelpad=30)
    ax[i].set_xlabel('redshift')
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    fig.savefig("{0}/features_{1}.png".format(fig_dir, sntype), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    fig_dir = os.path.join(ROOT_DIR, 'plasticc', 'Figures')
    feat_names = ['redshift', 'variance', 'skewness', 'kurtosis', 'amplitude', 'skew', 'somean', 'shapiro', 'q31',
                  'rms', 'mad', 'stetsonj', 'stetsonk', 'acorr', 'hlratio']
    plot_features(table_name='features', feature_names=feat_names, field='DDF', sntype=1, limit=100000,
                  fig_dir=fig_dir)
