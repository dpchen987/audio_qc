
import asyncio
import json
import websockets
from asr_api_server import config
from asr_api_server.logger import logger


WS_START = json.dumps({
    'signal': 'start',
    'nbest': 1,
    'continuous_decoding': False,
})
WS_END = json.dumps({
    'signal': 'end'
})


async def ws_rec(data):
    ws = config.get_ws()
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
        # this except has no effect
        logger.error(e)
    return ''.join(texts)

