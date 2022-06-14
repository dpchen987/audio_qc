#!/usr/bin/env python
# coding:utf-8


import os
import argparse


def get_ldd(bin_file):
    cmd = f'ldd {bin_file}'
    print(cmd)
    p = os.popen(cmd)
    out = p.read()
    lines = out.split('\n')
    ldds = []
    for line in lines:
        if '/home/' not in line:
            continue
        src = line.split('=>')[1].strip().split(' ')[0]
        ldds.append(src)
    return ldds


def main():
    parser = argparse.ArgumentParser(description='copy ldd of ELF')
    parser.add_argument('-s', '--src', help='source dir of binary to copy')
    parser.add_argument('-d', '--dst', help='destination dir for copying')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    runable = ['decoder_main', 'websocket_server_main']
    runable = [os.path.join(args.src, r) for r in runable]
    bins = ' '.join(runable)
    cmd = f'cp {bins} {args.dst}'
    print(cmd)
    os.system(cmd)
    srcs = set()
    for bin_file in runable:
        ldds = get_ldd(bin_file)
        srcs.update(ldds)
    srcs = ' '.join(srcs)
    cmd = f'cp {srcs} {args.dst}'
    print(cmd)
    os.system(cmd)
    print('done')


main()
