# !/usr/bin/env python
"""
Get PLASTICC data from SQL database
"""

import database


class GetData(object):

    def __init__(self, data_release):
        self.data_release = "release_{}".format(data_release)

    def get_object_ids(self):
        """ Get list of all object ids """
        obj_ids = database.exec_sql_query("SELECT objid FROM {0};".format(self.data_release))
        return obj_ids

    def get_light_curve(self, ptrobs_min, ptrobs_max):
        """ Get lightcurve from fits file """
        lc_data = None
        return lc_data

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
        lc_data : tuple
            A generator containing arrays in a tuple (mjd_date array, filter array, mag array, mag_err array)
        """

        result = database.exec_sql_query(
            "SELECT * FROM {0} WHERE objid LIKE '{1}%' AND objid LIKE '%{2}%' AND objid LIKE '%{3}%' AND objid LIKE '%{4}';".format(
                self.data_release, field, model, base, snid))

        num_lightcurves = len(result)
        for i in range(num_lightcurves):
            pass#lc_data = get_light_curve(ptrobs_min, ptrobs_max)

            yield result[i]#, lc_data


if __name__ == '__main__':
    getdata = GetData('20180112')
    result = getdata.get_transient_data(field='DDF', base='NONIa')
    print(next(result))
