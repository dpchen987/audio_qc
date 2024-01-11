import re
import json
import websockets
from asr_api_server import config
from asr_api_server.logger import logger
import time
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np

TRITON_FLAGS = {
    'url': "localhost:8001",
    'verbose': False,
    'model_name': 'infer_pipeline',
}

WS_START = json.dumps({
    'signal': 'start',
    'nbest': 1,
    'continuous_decoding': False,
})
WS_END = json.dumps({
    'signal': 'end'
})

FUNASR_WS_START = json.dumps({
    'chunk_size': [5, 10, 5],
    'wav_name': 'yshy',
    'is_speaking': True,
})

FUNASR_WS_END = json.dumps({
    'is_speaking': False
})


async def ws_rec(data):
    texts, message = '', ''
    for i in range(3):
        try:
            if config.CONF['decoder_server'] == 'funasr_websocket':
                texts = await funasr_websocket_rec(data)
            elif config.CONF['decoder_server'] == 'funasr_triton':
                texts = await funasr_triton_rec(data)
            elif config.CONF['decoder_server'] == 'wenet_websocket':
                texts = await wenet_websocket_rec(data)
            break
        except Exception as e:
            logger.debug(e)
            message = str(e)
            time.sleep(3)
    return texts, message



async def funasr_websocket_rec(data: bytes) -> str:
    """
    data : int16-bytes
    """
    funasr_websocket_uri = config.get_decoder_server_uri()
    conn = await websockets.connect(funasr_websocket_uri, ping_timeout=200, ssl=None)
    # step 1: send start
    await conn.send(FUNASR_WS_START)
    # step 2: send audio data
    await conn.send(data)
    # step 3: send end
    await conn.send(FUNASR_WS_END)
    res = await conn.recv()
    text = json.loads(res)['text']
    return text


async def funasr_triton_rec(data: bytes) -> str:
    """

    :param data: int16字节音频数据
    :return:
    """
    samples = np.frombuffer(data, dtype='int16')
    samples = np.array([samples], dtype=np.float32)
    lengths = np.array([[len(samples)]], dtype=np.int32)

    protocol_client = grpcclient
    inputs = [
        protocol_client.InferInput(
            "WAV", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        protocol_client.InferInput(
            "WAV_LENS", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
    ]
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]
    sequence_id = 10086

    triton_client = grpcclient.InferenceServerClient(
        url=config.get_decoder_server_uri(),
        verbose=TRITON_FLAGS['verbose']
    )
    response = await triton_client.infer(
        TRITON_FLAGS['model_name'],
        inputs,
        request_id=str(sequence_id),
        outputs=outputs,
    )
    text = response.as_numpy("TRANSCRIPTS")[0].decode("utf-8")
    # 替换语音片段merge时加的哈哈（被识别为呵呵）
    print(f'origin {text = }')
    text = re.sub(r'哈{2,8}|呵{2,8}|,{2,8}', ',', text)
    text = text.strip(',')
    # replace() is 9x faster than re.sub()
    # text = text.replace('哈哈', ',').replace('呵呵', ',')
    return text


async def wenet_websocket_rec(data):
    ws = config.get_decoder_server_uri()
    texts = []
    conn = await websockets.connect(ws, ping_timeout=200)
    # async with websockets.connect(ws) as conn:
    # step 1: send start
    await conn.send(WS_START)
    ret = await conn.recv()
    # step 2: send audio data
    await conn.send(data)
    # step 3: send end
    await conn.send(WS_END)
    # step 3: receive result
    i = 0
    while 1:
        i += 1
        ret = await conn.recv()
        ret = json.loads(ret)
        if ret['type'] == 'final_result':
            nbest = json.loads(ret['nbest'])
            text = nbest[0]['sentence']
            texts.append(text)
        elif ret['type'] == 'speech_end':
            break
    try:
        await conn.close()
    except Exception as e:
        # this except has no effect, just log as debug
        logger.debug(e)
    return ''.join(texts)
