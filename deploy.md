# 概述

ASR 服务包括两个：（docker 镜像同名）

1. asr_api_server 业务调用的API服务
    * 默认端口： 8300
2. asr_decode_server ASR转写服务
    * 默认端口： 8301


# 运行 asr_api_server:

```bash
docker run --net=host -e ASR_API_URL_DB=/app/url.db -v /opt/asr:/app -d --name asr_api_server --restart=always asr_api_server:0.6.2
```

# 运行 asr_decode_server:

```bash
docker run --net=host -d --name asr_decoder_server --restart=always asr_decode_server:0.6.2
```

# 运行参数

通过环境变量传递参数给服务，方便docker 容器运行时指定不同的参数。

| 环境变量            | 默认值      | 含义                                                                     |
| --                  | --          | --                                                                       |
| ASR_API_URL_DB      | ‘’          | 临时存储音频URL的leveldb路径，需保存在host文件系统，以防docker重启而丢失 |
| ASR_API_HOST        | 0.0.0.0     | host                                                                     |
| ASR_API_PORT        | 8300        | port                                                                     |
| ASR_API_CONCURRENCY | CPU核数x0.5 | 最大并发数                                                               |
| ASR_WS              | ws://127.0.0.1:8301        | asr_decode_server URI                                                    |
