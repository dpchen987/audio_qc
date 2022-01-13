#!/usr/bin/env python3

import requests


api = 'http://127.0.0.1:8300/asr/v1/rec'
# api = 'https://httpbin.org/post'

def test_one(audio_file):
    headers = {
        'appkey': '123',
        'format': 'pcm',
        'audio-url': 'https://yinshuhuiyuan-oss-10001.oss.jingan-hlw.inspurcloudoss.com/video/AHC0022101232683/908159455844241408/1636615795209.opus'
        #'Content-Type': 'application/octet-stream',
    }
    if 'audio-url' in headers:
        r = requests.post(api, headers=headers)
    else:
        with open(audio_file, 'rb') as f:
            data = f.read()
        r = requests.post(api, data=data, headers=headers)
    print(r.text)


if __name__ == '__main__':
    from sys import argv
    fn = argv[1]
    test_one(fn)
