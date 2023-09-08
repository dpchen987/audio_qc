version=$(cat ../asr_api_server/__init__.py | awk '{print $3}' | tr -d "'")
echo "version: $version"
docker run --net=host -d --name asr_decoder_server -v /usr/local/cuda:/usr/local/cuda --gpus all --restart=always asr_decode_server:$version