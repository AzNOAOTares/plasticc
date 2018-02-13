# -*- coding: UTF-8 -*-
"""
Entry point for plasticc module
"""

from __future__ import absolute_import
from __future__ import unicode_literals
import sys
# currently this is set to make_index, but eventually, we just want to argparse and run pipeline
from .make_index import main

if __name__=='__main__':
    main()
