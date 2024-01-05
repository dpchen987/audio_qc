#!/usr/bin/env python3

fn = '../asr_api_server/__init__.py'
version = '0.0.0'
with open(fn) as f:
    for line in f:
        if '__version__' in line:
            version = line.split('=')[-1].strip().strip('\'"')
            break
print(version)
