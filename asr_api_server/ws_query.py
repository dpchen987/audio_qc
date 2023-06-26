import json
import websockets
from asr_api_server import config
from asr_api_server.logger import logger
import time
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import np_to_triton_dtype
import asyncio
import soundfile as sf
import numpy as np
import os

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


async def ws_rec(data):
    texts = ''
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
            time.sleep(3)
    return texts


async def funasr_websocket_rec(data: bytes) -> str:
    funasr_websocket_uri = config.get_decoder_server_uri()
    server_ip, port = funasr_websocket_uri.split(':')
    funasr_wss_client = './funasr-wss-client'
    os.makedirs('.cache') if not os.path.exists('.cache') else None
    wav_file = '.cache//' + time.time().hex() + '.wav'
    waveform = np.frombuffer(data, dtype=np.int16)
    sf.write(wav_file, waveform, 16000)
    cmd_rec = [
        funasr_wss_client,
        f"--server-ip {server_ip}",
        f"--port {port}",
        f"--wav-path {wav_file}",
        f"--thread-num 1",
        f"--is-ssl 1",
    ]
    process = await asyncio.create_subprocess_exec(*cmd_rec, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    text = json.loads(stdout.decode('utf-8').strip())['text']
    os.remove(wav_file) if os.path.exists(wav_file) else None
    return text


async def funasr_triton_rec(data: bytes) -> str:
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

    return ''.join(results)


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
