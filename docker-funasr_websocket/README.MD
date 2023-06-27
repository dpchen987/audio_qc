# FunASR-websocket-docker

Service with websocket-cpp

- [FunASR-websocket官方说明](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/websocket/readme.md)
- version: 0.1

---


## docker编译镜像
```shell
docker build -t funasr_websocket_server:0.1 .
```


## docker启动服务示例
```shell
model_dir=/workspace/model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
vad_dir=/workspace/model/speech_fsmn_vad_zh-cn-16k-common-pytorch
punc_dir=/workspace/model/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
docker run --net=host \
  --rm \
  -v /aidata/junjie/repo/modelscope/:/workspace/model \
  -e MODEL_DIR=$model_dir \
  -e QUANTIZE=true \
  -e VAD_DIR=$vad_dir \
  -e VAD_QUANT=true \
  -e PUNC_DIR=$punc_dir \
  -e PUNC_QUANT=true \
  -e PORT=10095 \
  -e DECODER_THREAD_NUM=8 \
  -e IO_THREAD_NUM=8 \
  --name funasr_websocket_server \
  funasr_websocket_server:0.1   
```


## docker环境参数说明

- MODEL_DIR <string>: 默认值/workspace/models/asr, asr模型目录, 包含model.onnx, config.yaml, am.mvn；
- QUANTIZE <string>: 默认值true, load the model of model.onnx in model_dir. If set true, load the model of model_quant.onnx in model_dir
- VAD_DIR <string>: 默认值 /workspace/models/vad, the vad model path, which contains model.onnx, vad.yaml, vad.mvn
- VAD_QUANT <string>： 默认值true, load the model of model.onnx in vad_dir. If set true, load the model of model_quant.onnx in vad_dir
- PUNC_DIR <string>: 默认值/workspace/models/punc, the punc model path, which contains model.onnx, punc.yaml
- PUNC_QUANT <string>：默认值true, load the model of model.onnx in punc_dir. If set true, load the model of model_quant.onnx in punc_dir
- DECODER_THREAD_NUM <int>： 默认值8，number of threads for decoder；
- IO_THREAD_NUM <int>：默认值8，number of threads for network io；
- PORT <int>： 默认值10095，监听端口；
- CERTFILE <string>：默认值./server.crt, path of certficate for WSS connection. if it is empty, it will be in WS mode.
- KEYFILE <string>：默认值./server.key, path of keyfile for WSS connection


## Performance
- test_date: 2023/06/27
- test_server: algo-local-server
- test_model: [speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
- test_data: test-data/asr-test-3000-wav_scp.txt
- test_version: asr_api_server:0.8.0+funasr_websocket_server:0.1

| model_quant | concurrent-tasks | request_time(s) | RTF | Speed | WER |
| :----- | :----- | :------: | -----: | -----: | -----: |
| yes | 100 | 114  | 0.00697 | 143 | 4.26% |
| yes | 300 | 119  | 0.00723 | 138 | 4.26% |
| yes | 500 | 130  | 0.00792 | 126 | 4.26% |
| no | 100 | 152  | 0.00929 | 107 | 4.22% |
| no | 300 | 147  | 0.00899 | 111 | 4.22% |
| no | 500 | 148  | 0.00901 | 110 | 4.22% |