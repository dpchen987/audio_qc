# -*- coding: utf-8 -*-
'''
server的相关配置均由环境变量赋值，省去配置文件，方便部署
'''

import os
import re


CONF = dict(
    # for API server
    host='0.0.0.0',
    port=8300,
    workers=1,
    download_timeout=2,  # audio download timeout seconds
    vad_device='gpu',
    vad_max=15000,
    joiner=b'ailab@yshy',
    # for ASR processor
    # `wenet_websocket` or `funasr_triton` or `funasr_websocket`
    decoder_server='funasr_triton',
    decoder_server_uri=['127.0.0.1:8301'],
    redis_ip='127.0.0.1',
    redis_port=6379,
    concurrency=10,
    consumer='asr-c0',
    claim_min_idle_time=0,
)

decoder_types = ['wenet_websocket', 'funasr_triton', 'funasr_websocket']


def parse_env():
    global CONF
    CONF['vad_max'] = int(os.getenv('ASR_VAD_MAX', CONF['vad_max']))
    CONF['host'] = os.getenv('ASR_API_HOST', CONF['host'])
    CONF['port'] = int(os.getenv('ASR_API_PORT', CONF['port']))
    CONF['download_timeout'] = int(os.getenv('DOWNLOAD_TIMEOUT', CONF['download_timeout']))
    CONF['vad_device'] = os.getenv('VAD_DEVICE', CONF['vad_device'])
    CONF['workers'] = int(os.getenv('ASR_API_WORKERS', CONF['workers']))
    ds = os.getenv('ASR_DECODER_SERVER', CONF['decoder_server']).lower()
    assert ds in decoder_types, f'ASR_DECODER_SERVER should be {decoder_types}'
    CONF['decoder_server'] = ds
    decoder_server_uri = os.getenv('ASR_DECODER_SERVER_URI', '')
    if decoder_server_uri:
        uris = re.split(r'[,\s]+', decoder_server_uri)
        for uri in uris:
            if not re.search(r':\d+', uri):
                raise ValueError(f'Invalid decoder server uri {uri}.')
        CONF['decoder_server_uri'] = uris
    CONF['redis_ip'] = os.getenv('ASR_REDIS_IP', CONF['redis_ip'])
    CONF['redis_port'] = int(os.getenv('ASR_REDIS_PORT', CONF['redis_port']))
    CONF['concurrency'] = int(os.getenv('ASR_CONCURRENCY', 10))
    CONF['consumer'] = os.getenv('ASR_CONSUMER', CONF['consumer'])
    claim_min_idle_time = float(os.getenv('CLAIM_MIN_IDLE_TIME', 0))
    if claim_min_idle_time > 5:
        raise ValueError('claim_min_idle_time should be < 5 hours')
    # hour to milliseconds
    CONF['claim_min_idle_time'] = int(claim_min_idle_time * 3600 * 1000)
    print('*' * 66)
    print(f"ASR API URI: {CONF['host']}:{CONF['port']}")
    print('-' * 66)
    print(f"ASR Decoder Server: {CONF['decoder_server']}")
    print('-' * 66)
    print(f"ASR Decoder Server URI: {CONF['decoder_server_uri']}")
    print('*' * 66)
    print(f'{CONF = }')
    print('*' * 66)


parse_env()

DECODER_SERVER_INDEX = 0


def get_decoder_server_uri():
    global DECODER_SERVER_INDEX
    if len(CONF['decoder_server_uri']) == 1:
        return CONF['decoder_server_uri'][0]
    idx = DECODER_SERVER_INDEX % len(CONF['decoder_server_uri'])
    DECODER_SERVER_INDEX = idx + 1
    return CONF['decoder_server_uri'][idx]
