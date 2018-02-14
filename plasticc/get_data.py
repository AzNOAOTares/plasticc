# !/usr/bin/env python
"""
Get PLASTICC data from SQL database
"""
import os
import numpy as np
import astropy.io.fits as afits
import astropy.table as at
import database

ROOT_DIR = os.getenv('PLASTICC_DIR')
DATA_DIR = os.path.join(ROOT_DIR, 'plasticc_data')


class GetData(object):

    def __init__(self, data_release):
        self.data_release = "release_{}".format(data_release)
        self.phot_fields = ['MJD', 'FLT', 'MAG', 'MAGERR']

    def get_object_ids(self):
        """ Get list of all object ids """
        obj_ids = database.exec_sql_query("SELECT objid FROM {0};".format(self.data_release))
        return obj_ids

    def get_light_curve(self, objid, ptrobs_min, ptrobs_max):
        """ Get lightcurve from fits file """
        field, model, base, snid = objid.split('_')
        filename = "LSST_{0}_MODEL{1}/LSST_{0}_{2}_PHOT.FITS".format(field, model, base)
        phot_file = os.path.join(DATA_DIR, self.data_release.replace('release_', ''), filename)
        
        phot_HDU = afits.open(phot_file)
        phot_data = phot_HDU[1].data[ptrobs_min : ptrobs_max]

        phot_out = np.array([phot_data[field] for field in self.phot_fields])

        return phot_out

    def get_transient_data(self, field='%', model="%", base="%", snid="%"):
        """ Gets the light curve and header data given specific conditions. Returns a generator of LC info.

        Parameters
        ----------
        field : str, optional
            The field name. E.g. field='DDF' or field='WFD'. The default is '%' indicating that all fields will be included.
        model : str, optional
            The model number. E.g. model='04'. The default is '%' indicating that all model numbers will be included.
        base : str, optional
            The base name. E.g. base='NONIa'. The default is '%' indicating that all base names will be included.
        snid : str, optional
            The transient id. E.g. snid='87287'. The default is '%' indicating that all snids will be included.

        Return
        -------
        result: tuple
            A generator tuple containing (objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd)
        phot_data : numpy array
            A generator containing an array of numpy arrays [mjd_date array, filter array, mag array, mag_err array]
        """

        header = database.exec_sql_query(
            "SELECT * FROM {0} WHERE objid LIKE '{1}%' AND objid LIKE '%{2}%' AND objid LIKE '%{3}%' AND objid LIKE '%{4}';".format(
                self.data_release, field, model, base, snid))
        
        objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = list(zip(*header))
        num_lightcurves = len(objid)

        for i in range(num_lightcurves):
            phot_data = self.get_light_curve(objid[i], ptrobs_min[i], ptrobs_max[i])

            yield header[i], phot_data


if __name__ == '__main__':
    getdata = GetData('20180112')
    result = getdata.get_transient_data(field='DDF', base='NONIa')
    head, phot = next(result)
    # objid, ptrobs_min, ptrobs_max, mwebv, mwebv_err, z, zerr, sntype, peak_mjd = list(zip(*head))
    # mjd, filt, mag, mag_err = phot
    print(head, phot)
