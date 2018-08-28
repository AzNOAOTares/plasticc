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
import plasticc.database
import matplotlib.pyplot as plt 
import numpy as np 

def main():
    query = "SELECT ra, decl FROM release_20180827 WHERE ((sntype=51) or (sntype=151)) AND ((objid LIKE 'DDF%') or (objid LIKE 'WFD%'))"
    results = plasticc.database.exec_sql_query(query)
    ra, dec = zip(*results)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.scatter(ra, dec, marker='o')
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    fig.tight_layout(rect=[0,0,1,1])
    fig.savefig('kn_spatial_20180827.pdf')


if __name__=='__main__':
    sys.exit(main())
