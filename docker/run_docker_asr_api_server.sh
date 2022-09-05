version=$(cat ../asr_api_server/__init__.py | awk '{print $3}' | tr -d "'")
echo "version: $version"
docker run --net=host -e ASR_API_URL_DB=/app/url.db -v /path-in-host/:/app -d --name asr_api_server --restart=always asr_api_server:$version
