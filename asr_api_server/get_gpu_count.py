#!/usr/bin/env python
# coding:utf-8


import os


def get_gpu_count():
    cmd = 'nvidia-smi -q -d Memory'
    out = os.popen(cmd).read().split('\n')
    count = 0
    for line in out:
        if 'Attached GPUs' in line:
            count = int(line.split(':')[-1])
            break
    return count


if __name__ == '__main__':
    print(get_gpu_count())
