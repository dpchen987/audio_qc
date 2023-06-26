#!/bin/env python3


import os
import sys
import time


def get_version(fn):
    version = '0.0.0'
    with open(fn) as f:
        for line in f:
            if '__version__' in line:
                version = line.split('=')[-1].strip().strip('\'"')
                break
    return version


def main(save_image=False):
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} bin_dir model_dir')
        return
    bin_dir = sys.argv[1]
    model_dir = sys.argv[2]
    sub_dir = os.path.join(bin_dir, 'models')
    if os.path.exists(sub_dir):
        os.system(f'rm -rf {sub_dir}')
    cmd = f'cp -r {model_dir} {sub_dir}'
    print(cmd)
    os.system(cmd)
    os.system(f'cp run-decoder {bin_dir}')
    os.system(f'cp Dockerfile {bin_dir}')
    version = get_version('../asr_api_server/__init__.py')
    name = 'asr_decode_server'
    print(f'{name=}:{version=}')
    cmd = f'docker build -t {name}:{version} {bin_dir}'
    print(cmd)
    os.system(cmd)
    if save_image:
        now = time.strftime('%Y-%m-%d_%H-%M-%S')
        cmd = f'docker save -o image-{name}-{version}-{now}.tar {name}:{version}'
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    main()
