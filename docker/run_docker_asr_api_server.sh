version=$(cat ../asr_api_server/__init__.py | awk '{print $3}' | tr -d "'")
echo "version: $version"
docker run --net=host --gpus all -v /usr/local/cuda:/usr/local/cuda -e ASR_API_URL_DB=/app/url.db -e ASR_API_HOST=0.0.0.0 -e ASR_API_PORT=8400 -e ASR_WS=ws://127.0.0.1:8301 -d --name asr_api_server --restart=always asr_api_server:$version
