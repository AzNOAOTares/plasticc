#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')
sys.path.append(WORK_DIR)
import numpy as np
import plasticc
import plasticc.get_data
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib import pyplot as plt
from matplotlib.collections import EllipseCollection


def plot_corr_ellipses(data, ax=None, **kwargs):

    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError('data must be a 2D array')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
        ax.set_xlim(-0.5, M.shape[1] - 0.5)
        ax.set_ylim(-0.5, M.shape[0] - 0.5)

    # xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # set the relative sizes of the major/minor axes according to the strength of
    # the positive/negative correlation
    w = np.ones_like(M).ravel()
    h = 1 - np.abs(M).ravel()
    a = 45 * np.sign(M).ravel()

    ec = EllipseCollection(widths=w, heights=h, angles=a, units='x', offsets=xy,
                           transOffset=ax.transData, array=M.ravel(), **kwargs)
    ax.add_collection(ec)

    # if data is a DataFrame, use the row/column names as tick labels
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(M.shape[1]))
        ax.set_xticklabels(data.columns, rotation=30)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(data.index)
    return ec


def main():
    fig_dir = os.path.join(WORK_DIR, 'Figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    kwargs = plasticc.get_data.parse_getdata_options()
    print("This config ", kwargs)
    data_release = kwargs.pop('data_release')

    _ = kwargs.pop('model')
    field = kwargs.get('field')
    kwargs['columns']=['snid','ra', 'decl', 'hostgal_photoz', 'hostgal_photoz_err', 'mwebv', 'sntype' ]
    kwargs['model'] = '%'
    if field not in ('WFD', 'DDF'):
        message = 'Does not make sense to analyze DDF + WFD together. Specify one.'
        raise RuntimeError(message)
    kwargs['big'] = True
    kwargs['extrasql'] = 'AND sntype > 100'

    aggregate_map = plasticc.get_data.GetData.aggregate_sntypes()
    getter = plasticc.get_data.GetData(data_release)
    head = getter.get_lcs_headers(**kwargs)
    objid, ra, dec, hz, dhz, mwebv, sntype = zip(*list(head))
    target = [aggregate_map.get(x, 99) if x < 100 else aggregate_map.get(x-100, 99) for x in sntype] 
    target = np.array(target, dtype=np.int8)
    utypes = sorted(np.unique(target))
    new_types = np.arange(len(utypes))
    new_map = dict(zip(utypes, new_types))
    new_target = [new_map.get(x, 99) for x in target] 

    if field == 'DDF':
        data = {'objid':np.array(objid, dtype=np.int32), 'hostz':hz, 'hostz_err':dhz, 'mwebv':mwebv, 'target':new_target}
    else:
        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        l = c.galactic.l.value
        b = c.galactic.b.value
        data = {'objid':np.array(objid, dtype=np.int32), 'gall':l, 'galb':b, 'hostz':hz, 'hostz_err':dhz, 'mwebv':mwebv, 'target':target}

    df = pd.DataFrame.from_dict(data)
    df.sort_values('objid', inplace=True)
    df = df.assign(rowid=pd.Series(np.arange(len(objid))).values)

    corr_mat = df.corr()
    print(corr_mat)

    fig, ax = plt.subplots(1, 1)
    ec = plot_corr_ellipses(corr_mat, ax=ax, cmap='seismic', clim=[-1, 1])
    cb = fig.colorbar(ec)
    cb.set_label('Correlation coefficient')
    ax.margins(0.1)
    out_fn = f'header_correlations_{data_release}_{field}.pdf'
    out_fn = os.path.join(fig_dir, out_fn)
    fig.savefig(out_fn)

    #df.drop(['hostz_err','rowid'], inplace=True, axis=1)
    #g = sns.clustermap(data=df, annot=False, row_cluster=True, col_cluster=False, standard_scale=1, yticklabels=False, metric="euclidean", robust=True, figsize=(6, 12))
    #fig2 = g.fig
    #out_fn = out_fn.replace('correlations','clusters').replace('pdf','png')
    #fig2.savefig(out_fn)


if __name__=='__main__':
    sys.exit(main())
