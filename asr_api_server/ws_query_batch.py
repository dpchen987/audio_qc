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

WS_START = {
    'signal': 'start',
    'nbest': 1,
    'batch_lens': [],
    'enable_timestamp': False,
}

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
    texts, message = [], ""
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


async def funasr_websocket_rec(data: bytes) -> list:
    """
    data : int16-bytes
    """
    results = []
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
    results.append(text)
    return results


async def funasr_triton_rec(data: bytes) -> list:
    """

    :param data: int16字节音频数据
    :return:
    """
    results = []

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

    results.append(text)

    return results


async def wenet_websocket_rec(data):
    assert isinstance(data, list)
    ws = config.get_decoder_server_uri()
    conn = await websockets.connect(ws, ping_timeout=200)
    # async with websockets.connect(ws) as conn:
    # step 1: send start
    WS_START['batch_lens'] = [len(d) for d in data]
    await conn.send(json.dumps(WS_START))
    await conn.recv()
    # step 2: send audio data
    await conn.send(b''.join(data))
    result = await conn.recv()
    jn = json.loads(result)
    texts = []
    if jn['status'] != 'ok':
        print('failed from ws :', jn['message'])
    else:
        for result in jn['batch_result']:
            texts.append(result['nbest'][0]['sentence'])
    try:
        await conn.close()
    except Exception as e:
        # this except has no effect, just log as debug
        logger.debug(e)
    return texts
