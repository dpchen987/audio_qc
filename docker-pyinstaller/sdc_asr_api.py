#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import sys
import multiprocessing
from asr_api_server.main import run

if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(run())
