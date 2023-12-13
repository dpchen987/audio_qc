import time
import aiohttp

from .logger import logger


async def download(url, timeout_sec: int = 80, max_attempts: int = 3):
    b = time.time()
    attempt = 1
    download_success = False
    while attempt <= max_attempts:
        attempt_start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout_sec) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        msg = 'ok'
                        download_success = True
                        break
                    else:
                        data = b''
                        msg = f'Try {attempt} time download {url} failed with status: {resp.status}'
        except Exception as e:
            logger.exception(e)
            data = b''
            msg = f'download audio url failed with exception: {repr(e)}'
        attempt_cost_time = time.time() - attempt_start_time
        logger.debug(
            f"Try {attempt} Attempt || Attempt Download Cost Time(s): {attempt_cost_time} || Download Success: {download_success}　||　Download URL: {url}")
        attempt += 1
    time_cost = time.time() - b
    logger.debug(
        f'Finish Download || Download Cost Time(s): {time_cost} || Download Success: {download_success} || Download Total Attempts: {attempt if attempt == 1 else attempt - 1} || Download URL: {url}')
    return data, msg



