# -*- coding: utf-8 -*-
'''
server的相关配置均由环境变量赋值，省去配置文件，方便部署
'''

import os
import re
import leveldb

CONF = dict(
    host='0.0.0.0',
    port=8300,
    url_db='./db',  # must be set as environmente
    concurrency=10,
    decoder_server='funasr_websocket',  # `wenet_websocket` or `funasr_triton` or `funasr_websocket`
    # example: wenet_websocket_uri=['ws://127.0.0.1:8301'], funasr_triton_uri=['127.0.0.1:8001'], funasr_websocket_uri=['127.0.0.1:10095']
    decoder_server_uri=['127.0.0.1:10095'],
    download_timeout=20,  # audio download timeout seconds
    use_vad=False,  # whether use local vad
    vad_max=0,
    asr_response_delay=1.5, 
)


def parse_env():
    global CONF
    CONF['vad_max'] = int(os.getenv('ASR_VAD_MAX', CONF['vad_max']))
    CONF['asr_response_delay'] = float(os.getenv('ASR_RESPONSE_DELAY', CONF['asr_response_delay']))
    CONF['url_db'] = os.getenv('ASR_API_URL_DB', CONF['url_db'])
    if not CONF['url_db']:
        raise ValueError('environmente ASR_API_URL_DB must be set!!!')
    CONF['host'] = os.getenv('ASR_API_HOST', CONF['host'])
    CONF['port'] = int(os.getenv('ASR_API_PORT', CONF['port']))
    CONF['decoder_server'] = os.getenv('ASR_DECODER_SERVER', CONF['decoder_server']).lower()
    CONF['decoder_server_uri'] = os.getenv('ASR_DECODER_SERVER_URI', CONF['decoder_server_uri'])
    CONF['download_timeout'] = os.getenv('DOWNLOAD_TIMEOUT', CONF['download_timeout'])
    CONF['use_vad'] = bool(os.getenv('USE_VAD', CONF['use_vad']).lower() == 'true')

    assert CONF['decoder_server'] in ['wenet_websocket', 'funasr_triton', 'funasr_websocket'], \
        f'Invalid ASR_DECODER_SERVER: `{CONF["decoder_server"]}`, please input `wenet_websocket`, `funasr_triton` or `funasr_websocket`.'

    print('*' * 66)
    print(f"ASR API URI: {CONF['host']}:{CONF['port']}")
    print('-' * 66)
    print(f"ASR Decoder Server: {CONF['decoder_server']}")
    print('-' * 66)
    print(f"ASR Use VAD: {CONF['use_vad']}")
    print('-' * 66)

    # check decoder_server_uri
    decoder_server_uri = os.getenv('ASR_DECODER_SERVER_URI', '')
    if decoder_server_uri:
        uris = re.split(r'[,\s]+', decoder_server_uri)
        for uri in uris:
            if not re.search(r':\d+', uri):
                raise ValueError(f'Invalid decoder server uri {uri}.')
        CONF['decoder_server_uri'] = uris
    print(f"ASR Decoder Server URI: {CONF['decoder_server_uri']}")
    print('*' * 66)

    concur = os.getenv('ASR_API_CONCURRENCY', int(os.cpu_count() / 2))
    CONF['concurrency'] = int(concur)


parse_env()
WS_INDEX = 0
URL_INDEX = 0

DECODER_SERVER_INDEX = 0


def get_decoder_server_uri():
    global DECODER_SERVER_INDEX
    if len(CONF['decoder_server_uri']) == 1:
        return CONF['decoder_server_uri'][0]
    idx = DECODER_SERVER_INDEX % len(CONF['decoder_server_uri'])
    DECODER_SERVER_INDEX = idx + 1
    return CONF['decoder_server_uri'][idx]


url_db = leveldb.LevelDB(CONF['url_db'])

processing_set = set()
background_tasks = set()
