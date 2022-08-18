
import asyncio
import json
import websockets
from asr_api_server import config
from asr_api_server.logger import logger
from pprint import pprint


WS_START = {
    'signal': 'start',
    'nbest': 1,
    'batch_lens': [],
    'enable_timestamp': False,
}

WS_END = json.dumps({
    'signal': 'end'
})


async def ws_rec(data):
    assert isinstance(data, list)
    ws = config.get_ws()
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
