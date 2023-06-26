## asr_api_server

ASR解码服务docker-API

- 版本：0.8.0

---

## docker运行环境参数：

- **ASR_API_URL_DB**：数据库的url，默认值`./db`；
- **ASR_API_HOST**：api的ip地址，默认值`0.0.0.0`；
- **ASR_API_PORT**：api的端口，默认值`8300`；
- **ASR_API_CONCURRENCY**：api并发请求数，默认值CPU 数的一半；
- **ASR_DECODER_SERVER**：默认值`funasr_websocket`，ASR解码服务，可选`wenet_websocket` or `funasr_triton` or `funasr_websocket`
- **ASR_DECODER_SERVER_URI**：默认值：`127.0.0.1:10095`，解码服务的URI，如wenet_websocket_uri=`'ws://127.0.0.1:8301'`, funasr_triton_uri=`'127.0.0.1:8001'`, funasr_websocket_uri=`'127.0.0.1:10095'`，可用`, `隔开多次填写;
- **DOWNLOAD_TIMEOUT**：默认值`20`, 音频下载超时时间(s),会重试三次；
- **USE_VAD**：默认`False`，是否使用API中VAD;
- **ASR_VAD_MAX**：默认值`0`，API中vad最大数量限制，`0`为不限制。

## docker运行示例：

```shell
export version=0.8.0
export ASR_API_PORT=8301
export ASR_DECODER_SERVER_URI=127.0.0.1:10095
docker run --net=host \
  -it --rm \
  --gpus all \
  -v /usr/local/cuda:/usr/local/cuda \
  -e PATH=/usr/local/cuda/bin:$PATH \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  -e ASR_API_URL_DB=/app/url.db \
  -e ASR_API_HOST=0.0.0.0 \
  -e ASR_API_PORT=$ASR_API_PORT \
  -e ASR_DECODER_SERVER=funasr_websocket \
  -e ASR_DECODER_SERVER_URI=$ASR_DECODER_SERVER_URI \
  -e USE_VAD=false \
  --name asr_api_server_rm_$version_$ASR_API_PORT \
  asr_api_server:$version
```
  
## docker参数说明：

- `--net=host`：使用宿主机网络模式，使容器能够与宿主机共享网络。
- `--gpus all`：指定容器使用所有可用的GPU；指定设备索引为 0 的 GPU 设备运行，`--gpus device=0` 。
- `-e ASR_API_URL_DB=/app/url.db`：设置环境变量 `ASR_API_URL_DB` 为 `/app/url.db`，即 API URL 数据库的路径。
- `-e ASR_API_HOST=0.0.0.0`：设置环境变量 `ASR_API_HOST` 为 `0.0.0.0`，即 API Server 的主机地址。
- `-e ASR_API_PORT=8300`：设置环境变量 `ASR_API_PORT` 为 `8300`，即 API Server 的端口号。
- `-e ASR_WS=ws://127.0.0.1:8301`：设置环境变量 `ASR_WS` 为 `ws://127.0.0.1:8301`，即 Websocket Server 的地址；可设置多个`ASR_WS`地址，用`,\s `隔开,   如`-e ASR_WS='ws://127.0.0.1:8301, ws://127.0.0.1:8302'`。
- `-v `：将主机的cuda目录挂载到容器中。
- `-d`：在后台运行容器。
- `--name asr_api_server`：将容器命名为 `asr_api_server`。
- `--restart=always`：设置容器随着 Docker 的启动而自动重启。