#!/usr/bin/env python
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(WORK_DIR)
import numpy as np
import plasticc
import plasticc.get_data
import plasticc.database
import astropy.table as at
from collections import Counter

results = []
libs = []
obj = []
def main():
    dupes = at.Table.read('duplicated_rows.csv',format='csv')

    kwargs = plasticc.get_data.parse_getdata_options()
    data_release = kwargs.pop('data_release')
    for snid in dupes['object_id']:
        query = f'SELECT snid, sntype, sim_libid FROM release_{data_release} WHERE snid={snid}'
        result = plasticc.database.exec_sql_query(query)
        for r in result:
            if r[1] > 100:
                d = r[1] - 100
                libid = r[2]
                results.append(d)
                libs.append(libid)
                obj.append(snid)
                print(snid, d, libid)
            else:
                pass
    q = Counter(results)
    out = at.Table([obj, results, libs], names=['objid','snid','libid'])
    out.write('dupes_libid.csv', format='ascii.csv')

if __name__=='__main__':
    sys.exit(main())
