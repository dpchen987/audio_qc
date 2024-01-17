import json
import asyncio
import aiohttp

from .mqconn import mq
from .config import CONF
from .asr_funasr_triton import asr
from .logger import logger


TASKS = set()


async def post_data_to_url(url, data, retries=3):
    for _ in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=300) as response:
                    html = await response.text()
                    n = json.loads(html)
                    return n['code'] == 0
        except Exception as e:
            logger.error(f'Failed to post {url}, {e}')
            await asyncio.sleep(10)
    return False


async def rec(audio_data, retries=1):
    text, message = '', ''
    for _ in range(retries):
        try:
            text = await asr(audio_data)
            break
        except Exception as e:
            logger.error(e)
            message = str(e)
    return text, message


async def pipeline(mid, item):
    texts = []
    if item[b'data']:
        segments = item[b'data'].split(CONF['joiner'])
        for seg in segments:
            text, error = await rec(seg)
            if error:
                logger.error(f'pipeline rec() got error : {error}')
                return
            texts.append(text)
    else:
        logger.info('========= pipeline got empty audio =========')
    text = ','.join(texts)
    result = {
        'task_id': item[b'task_id'].decode('utf8'),
        'code': 0,
        'err_msg': 'success',
        'content': text,
    }
    good = await post_data_to_url(item[b'url'].decode('utf8'), result)
    if good:
        await mq.ack(mid)


async def once(wait=False):
    if not mq.conn:
        await mq.init()
    messages = await mq.pop(
        CONF['concurrency'],
        consumername=CONF['consumer'],
        block=200,
    )
    from_claim = False
    if not messages and CONF['claim_min_idle_time'] > 0:
        # logger.info('no message from mq, try to claim...')
        from_claim = True
        messages = await mq.claim(
            CONF['concurrency'],
            'asr-claim',
            CONF['claim_min_idle_time'],
        )
    if not messages:
        return
    logger.info(f'got mq {len(messages) = }, {from_claim = }')
    for m in messages:
        mid = m[0]
        task = asyncio.create_task(pipeline(mid, m[1]))
        task.add_done_callback(TASKS.discard)
        TASKS.add(task)
        await asyncio.sleep(0.01)
    if wait:
        while TASKS:
            await asyncio.sleep(0.1)


async def loop():
    await mq.init()
    while True:
        await once()
        await asyncio.sleep(1)


def main():
    print('\n********** asr worker starts **************', flush=True)
    asyncio.run(loop())


if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        asyncio.run(once(True))
    else:
        main()
