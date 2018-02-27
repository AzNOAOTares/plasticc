# !/usr/bin/env python
"""
Get PLASTICC data from SQL database
"""
import os
import numpy as np
import warnings
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


    def get_light_curve(self, objid, ptrobs_min, ptrobs_max, standard_zpt=30.):
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
                if pfield=='ZEROPT':
                    phot_dict[f][pfield] = np.repeat(standard_zpt, len(fIndexes))
                elif pfield=='FLUXCAL':
                    true_zpt = phot_data['ZEROPT'][fIndexes]
                    fluxcal  = (10**(-0.4*(true_zpt - standard_zpt)))*phot_data[pfield][fIndexes]
                    phot_dict[f][pfield] = fluxcal
                elif pfield=='FLUXCALERR':
                    true_zpt = phot_data['ZEROPT'][fIndexes]
                    fluxcalerr = (10**(-0.4*(true_zpt - standard_zpt)))*phot_data[pfield][fIndexes]
                    phot_dict[f][pfield] = fluxcalerr
                elif pfield=='MJD':
                    # MJD
                    phot_dict[f][pfield] = phot_data[pfield][fIndexes]
                elif pfield=='FLT':
                    true_zpt = phot_data['ZEROPT'][fIndexes]
                    nobs = len(true_zpt)
                    phot_dict[f][pfield] = np.repeat(f.strip(), nobs)
                else:
                    phot_dict[f][pfield] = phot_data[pfield][fIndexes]

        phot_out = pd.DataFrame(phot_dict)
        return phot_out

    @staticmethod
    def convert_pandas_lc_to_recarray_lc(phot):
        """
        ANTARES_object not Pandas format broken up by passband
        TODO: This is ugly - just have an option for get_lcs_data to return one or the other
        """
        pbs = ('u', 'g', 'r', 'i', 'z', 'Y')
        mjd   = []
        flux  = []
        dflux = []
        zpt   = []
        pb    = []

        for this_pb in phot:

            # do we know what this passband is
            if this_pb not in pbs:
                continue 

            this_pb_lc = phot.get(this_pb)
            if this_pb_lc is None:
                continue

            mjd   += this_pb_lc['MJD'].tolist()
            flux  += this_pb_lc['FLUXCAL'].tolist()
            dflux += this_pb_lc['FLUXCALERR'].tolist()
            pb    += this_pb_lc['FLT'].tolist()
            zpt   += this_pb_lc['ZEROPT'].tolist()

        out = np.rec.fromarrays([mjd, flux, dflux, pb, zpt], names=['mjd', 'flux', 'dflux', 'pb', 'zpt'])
        return out



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


    def get_lcs_headers(self, columns=None, field='%', model='%', base='%', snid='%', sntype='%', 
            get_num_lightcurves=False, limit=None, shuffle=False, sort=True, offset=0):
        """ Gets the header data given specific conditions.

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
        limit : int, optional 
            Limit the results to this number (> 0)
        shuffle : bool, optional
            Randomize the order of the results - not allowed with `sort`
        sort : bool, optional
            Order the results by objid - overrides `shuffle` if both are set
        offeet : int, optional
            Start returning MySQL results from this row number offset
        Return
        -------
        result: tuple
        """
        if columns is None:
            columns=['objid', 'ptrobs_min', 'ptrobs_max']

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

        if sort is True and shuffle is True:
            message = 'Cannot sort and shuffle at the same time! That makes no sense!'
            shuffle = False
            warnings.warn(message, RuntimeWarning)


        shuffle_command = '' if shuffle is False else " ORDER BY RAND()"
        sort_command  = '' if sort is False else ' ORDER BY objid'
        extra_command = ''.join([sntype_command, sort_command, shuffle_command, limit_command, offset_command])

        query = "SELECT {0} FROM {1} WHERE objid LIKE '{2}%' AND objid LIKE '%{3}%' AND objid LIKE '%{4}%' AND objid LIKE '%{5}' {6};".format(', '.join(columns), self.data_release, field, model, base, snid, extra_command)
        header = database.exec_sql_query(query)

        num_lightcurves = len(header)
        if get_num_lightcurves:
            return num_lightcurves

        if num_lightcurves > 0:
            return header
        else:
            print("No light curves in the database satisfy the given arguments. "
                  "field: {}, model: {}, base: {}, snid: {}, sntype: {}".format(field, model, base, snid, sntype))
            return []



    def get_lcs_data(self, columns=None, field='%', model='%', base='%', snid='%', sntype='%',\
            limit=None, shuffle=False, sort=True, offset=0):
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
        limit : int, optional 
            Limit the results to this number (> 0)
        shuffle : bool, optional
            Randomize the order of the results - not allowed with `sort`
        sort : bool, optional
            Order the results by objid - overrides `shuffle` if both are set
        offset : int, optional
            Start returning MySQL results from this row number offset (> 0)

        Return
        -------
        result: tuple
            A generator tuple containing (objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd)
        phot_data : pandas DataFrame
            A generator containing a DataFrame with the MJD, FLT, MAG, MAGERR as rows and the the filter names as columns.
            E.g. Access the magnitude in the z filter with phot_data['z']['MAG'].
        """

        header = self.get_lcs_headers(columns=columns, field=field,\
                    model=model, base=base, snid=snid, sntype=sntype,\
                    limit=limit, sort=sort, shuffle=shuffle, offset=offset)

        num_lightcurves = len(header)
        for i in range(num_lightcurves):
            objid, ptrobs_min, ptrobs_max = header[i][0:3]
            phot_data = self.get_light_curve(objid, ptrobs_min, ptrobs_max)
            yield header[i], phot_data



if __name__ == '__main__':
    getdata = GetData('20180112')
    result = getdata.get_transient_data(field='DDF', base='NONIa')
    head, phot = next(result)
    # objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = list(zip(*head))
    # mjd, filt, mag, mag_err = phot
    print(head, phot)



