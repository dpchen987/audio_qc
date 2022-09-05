import os
import asyncio
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import HTMLResponse

from asr_api_server.routers import router
from asr_api_server.logger import logger
from asr_api_server.config import CONF
from asr_api_server.asr_consumer import asy_timer
import leveldb

from fastapi.staticfiles import StaticFiles
root_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(root_dir, 'static')


def create_app():
    root_path = os.environ.get('ai_root_path', '')
    if root_path:
        logger.info("FastAPI running on root_path: {}".format(root_path))
    app = FastAPI(
        root_path=root_path
    )
    app.include_router(router, prefix="/asr/v1")
    return app


app = create_app()

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asy_timer())

@app.get("/test")
async def test():
    html_path = os.path.join(root_dir, 'templates/test.html')
    with open(html_path) as f:
        html = f.read()
    return HTMLResponse(content=html)


def run():
    import uvicorn
    reload = False
    uvicorn.run('asr_api_server.main:app', host=CONF['host'], port=CONF['port'], reload=reload, loop='uvloop')


if __name__ == '__main__':
    run()
