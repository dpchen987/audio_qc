#!/usr/bin/env python3

import os
import time
import requests

appkey = 'tHUd6PliZqHRki6h'
access_token = '2717331ce7484627957b325ab4fe12ec'


def rec(data):
    url = 'https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr'
    params = {
        'appkey': appkey,
        'format': 'pcm',
        'sample_rate': 16000,
        'enable_punctuation_prediction': True,
        'enable_inverse_text_normalization': True,
    }
    headers = {
        'X-NLS-Token': access_token,
        'Content-type': 'application/octet-stream',
        'Content-Length': str(len(data)),
        'Host': 'nls-gateway.cn-shanghai.aliyuncs.com'
    }
    b = time.time()
    r = requests.post(url, params=params, headers=headers, data=data)
    # print('time used:', time.time() - b)
    print(r.text)
    text = r.json()['result']
    # with open(fn + '.txt', 'w') as f:
    #     f.write(text)
    return text


def main(path_scp, path_trans):
    wavs = []
    with open(path_scp) as f:
        for l in f:
            zz = l.strip().split()
            if len(zz) != 2:
                print('invalid line:', l)
                continue
            key, audio_path = zz
            wavs.append((key, audio_path))
    print('wavs:', len(wavs))
    trans = open(path_trans, 'w+')
    b = time.time()
    print('start @', b)
    for key, wav in wavs:
        with open(wav, 'rb') as f:
            audio_data = f.read()
        text = rec(audio_data)
        print(key, '==>', text)
        trans.write(f'{key}\t{text}\n')
        trans.flush()
    e = time.time()
    trans.close()
    print('done @', e)
    print('time ', e - b)



if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.stderr.write(f"{sys.argv[0]} <in_scp> <out_trans>\n")
        exit(-1)
    path_scp = sys.argv[1]
    path_trans = sys.argv[2]
    main(path_scp, path_trans)
