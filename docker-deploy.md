# ASR-Docker部署

asr服务部署说明

---



## asr_api_server

- 版本：0.7.0

- 运行系统变量配置：

  - **ASR_API_URL_DB**：数据库的url，默认值`./db`；
  - **ASR_API_HOST**：api的ip地址，默认值`0.0.0.0`；
  - **ASR_API_PORT**：api的端口，默认值`8300`；
  - **ASR_WS**：WebSocket服务地址，可设置多个`ASR_WS`地址，用`,\s `隔开,   如`ASR_WS='ws://127.0.0.1:8301, ws://127.0.0.1:8302`。默认值`ws://127.0.0.1:8301`；
  - **ASR_API_CONCURRENCY**：api并发请求数，默认值CPU 数的一半；
  - **ASR_API_BACKEND**：ASR后端，可选`wenet`、`triton`，默认值`wenet`；
  - **ASR_URL**：Triton Inference Server的gRPC地址，可设置多个`ASR_URL`地址，用`,\s `隔开,   如`ASR_URL='127.0.0.1:8001, 127.0.0.1:8002'`。默认值`127.0.0.1:8001`；

- docker运行：
  - 命令行：

    ```shell
    docker run --net=host \
        -d \
        --gpus all \
        -v /usr/local/cuda:/usr/local/cuda \
        -e PATH=/usr/local/cuda/bin:$PATH \
        -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
        -e ASR_API_URL_DB=/app/url.db \
        -e ASR_API_HOST=0.0.0.0 \
        -e ASR_API_PORT=8300 \
        -e ASR_WS=ws://127.0.0.1:8301 \
        -e ASR_URL=127.0.0.1:8001 \
        -e ASR_API_BACKEND=wenet \
        --name asr_api_server \
        --restart=always \
        asr_api_server:0.6.7
    ```
  
  - 参数说明：
  
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
    
      



## triton_funasr

- 版本：2.2

- 运行系统变量配置：

  - **MODEL_REPOSITORY**：模型位置，需要挂载到`/workspace/`，默认值：`/workspace/model_repo_paraformer_large_offline`；
  - **PINNED_MEMORY_POOL_BYTE_SIZE**：设置固定内存池的字节大小，这个固定内存池用于在推理时存储模型参数和输入数据，它是通过锁定物理内存页来实现，可以提高内存访问性能并减少延迟，默认值`512000000`；
  - **CUDA_MEMORY_POOL_BYTE_SIZE**：设置 CUDA 内存池的字节大小，这个 CUDA 内存池用于在推理时分配和释放 CUDA 设备内存，它可以提高内存分配性能并减少延迟，避免了为每个推理请求分配和释放 CUDA 设备内存的开销，默认值`1024000000`；
  - **GRPC_PORT**：用于设置 gRPC 服务的监听端口，默认值`8001`；
  - **METRICS_PORT**：在 `METRICS_PORT` 端口上提供 Prometheus 监控指标，以便于监视 Triton Inference Server的性能指标和状态信息；默认值`8002`；
  - **HTTP_PORT**：用于设置 HTTP REST 服务的监听端口。当 Triton Inference Server 启动时，它会监听 `HTTP_PORT` 端口，等待客户端连接并处理推理请求；默认值`8301`。

- docker运行：

  - 说明：需要同时挂载cuda-11、cuda-12库及模型。

  - 命令行：

    ```sh
    docker run -d \
        --name "triton_funasr" \
        --net=host \
        --gpus all \
        --shm-size 1g \
        -v /usr/local/cuda-12.1:/usr/local/cuda-12 \
        -e PATH=/usr/local/cuda/bin:$PATH \
        -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
        -v /usr/local/cuda-11.4:/usr/local/cuda-11 \
        -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
        -v <path_host/model_repo_paraformer_large_offline>:/workspace/
        triton_funasr:2.2
    ```

  - 参数说明：

    - `--shm-size 1g`：用来设置容器内部的共享内存文件系统的大小，以 GB 为单位，默认值`1g`。



## asr_decode_server

- 版本：0.6.6
  
- 运行系统变量配置：
  
  - **ASR_WS_PORT**：WebSocket服务端口，默认8301；
  - **ASR_WS_CHUNK_SIZE**：传输数据块大小，默认-1；
  - **ASR_WS_CONTEXT_SCORE**：上下文相关性分数，默认8。
  
- docker运行：

  - 命令行：

    ```shell
    docker run --net=host \
        -d \
        -e ASR_WS_PORT=8301 \
        --gpus all \
        -v /usr/local/cuda:/usr/local/cuda \
        --name asr_decoder_server \
        --restart=always \
        asr_decode_server:0.6.6
    ```
  
  - 参数说明：
  
    - `--net=host`：指定容器使用主机网络。
    - `-d`：指定容器在后台运行。
    - `--name`：指定容器的名称。
    - `-v`：将主机的cuda目录挂载到容器中。
    - `--gpus all`：指定容器使用所有可用的GPU；指定设备索引为 0 的 GPU 设备运行，`--gpus device=0` 。
    - `-e`：设置环境变量，例如服务端口等。
    - `--restart=always`：设置容器随Docker自动重启。
  
  