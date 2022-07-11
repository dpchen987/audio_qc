version=$(cat ../asr_api_server/__init__.py | awk '{print $3}' | tr -d "'")
echo "version: $version"
docker run --net=host -e WEB_CONCURRENCY=2 -d --name asr_api_server --restart=always asr_api_server:$version
