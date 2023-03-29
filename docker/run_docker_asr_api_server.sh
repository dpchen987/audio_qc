version=$(cat ../asr_api_server/__init__.py | awk '{print $3}' | tr -d "'")
echo "version: $version"
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
  asr_api_server:$version
