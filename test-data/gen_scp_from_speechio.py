#!/usr/bin/env python
# coding:utf-8
'''
用于从https://github.com/SpeechColab/Leaderboard
提取生成测试集
'''


import random
import os


def main(data_dir, each_count, save_name):
    # _dir : path-to-Leaderboard/datasets
    subs = os.listdir(data_dir)
    result = []
    for s in subs:
        if s.startswith('SPEECHIO') or s.startswith('AISHELL'):
            meta = os.path.join(data_dir, s, 'metadata.tsv')
            with open(meta) as f:
                line1 = f.readline()  # skip title
                print('skip:', line1)
                lines = f.readlines()
            dirname = os.path.dirname(os.path.abspath(meta))
            if each_count:
                zz = random.sample(lines, each_count)
            else:
                # all
                zz = lines
            zz = [(dirname, z) for z in zz]
            result.extend(zz)
    scps = []
    transs = []
    has = set()
    for dirname, l in result:
        zz = l.strip().split()
        key = zz[0]
        if key in has:
            continue
        has.add(key)
        audio_path = os.path.join(dirname, zz[1])
        text = zz[-1]
        scps.append(f'{key}\t{audio_path}\n')
        transs.append(f'{key}\t{text}\n')
    print(f'{len(scps)=}, {len(transs)=}')
    with open(f'{save_name}-{len(scps)}-wav_scp.txt', 'w') as f:
        f.write(''.join(scps))
    with open(f'{save_name}-{len(scps)}-trans.txt', 'w') as f:
        f.write(''.join(transs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='select data for test-set')
    parser.add_argument('-c', '--count', type=int, required=True, help="select count from each set")
    parser.add_argument('-d', '--dir', required=True, help="dir to Leaderboard/datasets")
    parser.add_argument('-s', '--save', required=True, help="file path of wav_scp to save")
    args = parser.parse_args()
    main(args.dir, args.count, args.save)

