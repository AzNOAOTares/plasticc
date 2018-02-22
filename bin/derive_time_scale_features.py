#!/usr/bin/env python
import sys
import os
ROOT_DIR = os.getenv('PLASTICC_DIR')
WORK_DIR = os.path.join(ROOT_DIR, 'plasticc')
sys.path.append(WORK_DIR)
import ANTARES_object
import plasticc
import plasticc.database
import plasticc.get_data

def main():

    data_release = '20180221' # need to make this argparseable
    model_name = 'RRLyrae'

    getter = plasticc.get_data.GetData(data_release)
    sntypes_map = getter.get_sntypes()
    model_id = list(sntypes_map.keys())[list(sntypes_map.values()).index(model_name)]

    lcdata = getter.get_transient_data(model=model_id, sntype=model_id)
    if lcdata is None:
        raise RuntimeError('could not get light curvrs')
    else:
        for lc in lcdata:
            print(lc)

if __name__=='__main__':
    sys.exit(main())
