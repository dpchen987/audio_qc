import asyncio
import aiohttp
import time
import json, requests
import base64
import os
import random

api = 'http://192.168.10.10:8002/asr/v1/speech_vad'

file_ls = []
for file_name in list(os.walk('/aidata/audio/ahc_项目_办公室录音'))[0][-1]:
    if file_name.endswith(".opus"):
        audio_pth = "/aidata/audio/ahc_项目_办公室录音/" + file_name
        file_ls.append(audio_pth)

print( "files:", len(file_ls) )

async def fetch_url(para):
    f_no = para["task_id"]
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api, json=para) as response:
            end_time = time.time()
            print(f"Time taken to fetch {f_no}: {end_time - start_time:.2f} seconds")
            return await response.text()


async def main():
    tasks = []
    for f_no in range(1000):
        audio_pth = random.choice(file_ls)
        await asyncio.sleep(0.1)
        with open(audio_pth, 'rb') as fin:
            audio_data = fin.read()
        bs64 = base64.b64encode(audio_data).decode('latin1')
        para = {
          "task_id": f_no,
          "enable_punctution_prediction": True,
          "file_path": "http://ahc-audio.oss-cn-shanghai.aliyuncs.com/video/AHC0022101230113/1006728871187451904/1660120281154.opus",
          "callback_url": "http://localhost:8305/asr/v1/callBack_test",
          "file_content": json.dumps(bs64),
          "file_type": "opus",
          "trans_type": 2
        }
        tasks.append(asyncio.ensure_future(fetch_url(para)))
    responses = await asyncio.gather(*tasks)
    return responses

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(main())