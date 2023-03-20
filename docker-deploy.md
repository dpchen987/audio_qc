# ASR-Docker部署

asr服务部署说明

---

## asr_decode_server

- 运行系统变量配置：
  - **ASR_WS_PORT**：WebSocket服务端口，默认8301；
  - **ASR_WS_CHUNK_SIZE**：传输数据块大小，默认-1；
  - **ASR_WS_CONTEXT_SCORE**：上下文相关性分数，默认8。

- docker运行：

  - 命令行：

    ```shell
    $ docker run --net=host -d --name asr_decoder_server -e ASR_WS_PORT=8301 -v /usr/local/cuda:/usr/local/cuda --gpus all --restart=always asr_decode_server:0.6.6
    ```

  - 参数解释

    - `--net=host`指定容器使用主机网络。
    - `-d`指定容器在后台运行。
    - `--name`指定容器的名称。
    - `-v`将主机的cuda目录挂载到容器中。
    - `--gpus all`指定容器使用所有可用的GPU。
    - `-e`设置环境变量，例如服务端口等。
    - `--restart=always`设置容器随Docker自动重启。

  

## asr_api_server

- 运行系统变量配置：

  - **ASR_API_URL_DB**：数据库的url，默认值`./db`；
  - **ASR_API_HOST**：api的ip地址，默认值`0.0.0.0`；
  - **ASR_API_PORT**：api的端口，默认值`8300`；
  - **ASR_WS**：websocket server地址，默认值`''`；
  - **ASR_API_CONCURRENCY**：api并发请求数，默认值CPU 数的一半；

- docker运行：
  - 命令行：

    ```shell
    $ docker run --net=host --gpus all -e ASR_API_URL_DB=/app/url.db -e ASR_API_HOST=0.0.0.0 -e ASR_API_PORT=8300 -e ASR_WS='ws://127.0.0.1:8301, ws://127.0.0.1:8302' -d --name asr_api_server --restart=always asr_api_server:0.6.6
    ```

  - 参数说明：
  
    - `--net=host`：使用宿主机网络模式，使容器能够与宿主机共享网络。
  
    - `--gpus all`：指定容器使用所有可用的GPU。
  
    - `-e ASR_API_URL_DB=/app/url.db`：设置环境变量 `ASR_API_URL_DB` 为 `/app/url.db`，即 API URL 数据库的路径。
  
    - `-e ASR_API_HOST=0.0.0.0`：设置环境变量 `ASR_API_HOST` 为 `0.0.0.0`，即 API Server 的主机地址。
  
    - `-e ASR_API_PORT=8300`：设置环境变量 `ASR_API_PORT` 为 `8300`，即 API Server 的端口号。
  
    - `-e ASR_WS=ws://127.0.0.1:8301`：设置环境变量 `ASR_WS` 为 `ws://127.0.0.1:8301`，即 Websocket Server 的地址，可设置多个`ASR_WS`地址。
  
    - `-v `：将主机上的 `/path-in-host/` 目录挂载到容器内的 `/app` 目录中。此处可以根据需求进行修改。
  
    - `-d`：在后台运行容器。
  
    - `--name asr_api_server`：将容器命名为 `asr_api_server`。
  
    - `--restart=always`：设置容器随着 Docker 的启动而自动重启。
    
      
