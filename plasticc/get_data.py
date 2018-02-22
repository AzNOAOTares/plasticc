# !/usr/bin/env python
"""
Get PLASTICC data from SQL database
"""
import os
import numpy as np
import pandas as pd
import astropy.io.fits as afits
from . import database

ROOT_DIR = os.getenv('PLASTICC_DIR')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')


class GetData(object):

    def __init__(self, data_release):
        self.data_release = "release_{}".format(data_release)
        self.phot_fields = ['MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'ZEROPT']

    def get_object_ids(self):
        """ Get list of all object ids """
        obj_ids = database.exec_sql_query("SELECT objid FROM {0};".format(self.data_release))
        return obj_ids

    def get_column_for_sntype(self, column_name, sntype, field='%'):
        """ Get an sql column for a particular sntype class

        Parameters
        ----------
        column_name : str
            column name. E.g. column_name='peakmjd'
        sntype : int
            sntype number. E.g. sntype=4
        field : str, optional
            The field name. E.g. field='DDF' or field='WFD'. The default is '%' indicating that all fields will be included.

        Return
        -------
        column_out: list
            A list containing all the entire column for a particular sntype class
        """
        try:
            column_out = database.exec_sql_query("SELECT {0} FROM {1} WHERE objid LIKE '{2}%' AND sntype={3};".format(column_name, self.data_release, field, sntype))
            column_out = np.array(column_out)[:, 0]
        except IndexError:
            print("No data in the database satisfy the given arguments. field: {}, sntype: {}".format(field, sntype))
            return []
        return column_out


    def get_light_curve(self, objid, ptrobs_min, ptrobs_max):
        """ Get lightcurve from fits file

        Parameters
        ----------
        objid : str
            The object ID. E.g. objid='DDF_04_NONIa-0004_87287'
        ptrobs_min : int
            Min index of object in _PHOT.FITS.
        ptrobs_max : int
            Max index of object in _PHOT.FITS.

        Return
        -------
        phot_out: pandas DataFrame
            A DataFrame containing the MJD, FLT, FLUXCAL, FLUXCALERR, ZEROPT seperated by each filter.
            E.g. Access the magnitude in the z filter with phot_out['z']['MAG'].
        """
        field, model, base, snid = objid.split('_')
        filename = "LSST_{0}_MODEL{1}/LSST_{0}_{2}_PHOT.FITS".format(field, model, base)
        phot_file = os.path.join(DATA_DIR, self.data_release.replace('release_', ''), filename)
        if not os.path.exists(phot_file):
            phot_file = phot_file +'.gz'

        try:
            phot_HDU = afits.open(phot_file)
        except Exception as e:
            message = f'Could not open photometry file {phot_file}'
            raise RuntimeError(message)

        phot_data = phot_HDU[1].data[ptrobs_min-1:ptrobs_max]

        phot_dict = {}
        filters = list(set(phot_data['FLT']))  # e.g. ['i', 'r', 'Y', 'u', 'g', 'z']
        for f in filters:
            fIndexes = np.where(phot_data['FLT'] == f)[0]
            phot_dict[f] = {}
            for pfield in self.phot_fields:
                phot_dict[f][pfield] = phot_data[pfield][fIndexes]
        phot_out = pd.DataFrame(phot_dict)

        return phot_out

    def get_sntypes(self):
        sntypes_map = {1: 'SN1a', 2: 'CC', 3: 'SNIbc', 4: 'IIn', 42: 'SNIa-91bg', 45: 'pointIa', 50: 'Kilonova',
                        60: 'Magnetar', 61: 'PISN', 62: 'ILOT', 63: 'CART', 80: 'RRLyrae', 81: 'Mdwarf', 82: 'Mira',
                        90:'BSR', 91: 'String'}
        return sntypes_map

    def get_avail_sntypes(self):
        """ Returns a list of the different transient classes in the database. """
        sntypes = database.exec_sql_query("SELECT DISTINCT sntype FROM {};".format(self.data_release))
        sntypes_map = self.get_sntypes()
        return sorted([sntype[0] for sntype in sntypes]), sntypes_map

    def get_transient_data(self, columns=None, field='%', model='%', base='%', snid='%', sntype='%', get_num_lightcurves=False):
        """ Gets the light curve and header data given specific conditions. Returns a generator of LC info.

        Parameters
        ----------
        columns : list
            A list of strings of the names of the columns you want to retrieve from the database.
            You must at least include ['objid', 'ptrobs_min', 'ptrobs_max'] at the beginning of the input list.
            E.g. columns=['objid', 'ptrobs_min', 'ptrobs_max', 'sntype', 'peakmjd'].
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
        get_num_lightcurves : boolean, optional
            If this is True, then the return value is just a single iteration generator stating the number of
            light curves that satisfied the given conditions.

        Return
        -------
        result: tuple
            A generator tuple containing (objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd)
        phot_data : pandas DataFrame
            A generator containing a DataFrame with the MJD, FLT, MAG, MAGERR as rows and the the filter names as columns.
            E.g. Access the magnitude in the z filter with phot_data['z']['MAG'].
        """
        if columns is None:
            columns=['objid', 'ptrobs_min', 'ptrobs_max']
        sntype_command = '' if sntype == '%' else " AND sntype={}".format(sntype)
        header = database.exec_sql_query(
            "SELECT {0} FROM {1} WHERE objid LIKE '{2}%' AND objid LIKE '%{3}%' AND objid LIKE '%{4}%' "
            "AND objid LIKE '%{5}' {6};".format(', '.join(columns), self.data_release, field, model, base, snid, sntype_command))
        num_lightcurves = len(header)
        if get_num_lightcurves:
            yield num_lightcurves
            return
        try:
            for i in range(num_lightcurves):
                objid, ptrobs_min, ptrobs_max = header[i][0:3]
                phot_data = self.get_light_curve(objid, ptrobs_min, ptrobs_max)
                yield header[i], phot_data

        except ValueError:
            print("No light curves in the database satisfy the given arguments. "
                  "field: {}, model: {}, base: {}, snid: {}, sntype: {}".format(field, model, base, snid, sntype))
            return


if __name__ == '__main__':
    getdata = GetData('20180112')
    result = getdata.get_transient_data(field='DDF', base='NONIa')
    head, phot = next(result)
    # objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = list(zip(*head))
    # mjd, filt, mag, mag_err = phot
    print(head, phot)



