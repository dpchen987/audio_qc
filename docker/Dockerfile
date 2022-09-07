FROM python:3.8-slim
ARG version
WORKDIR /app
COPY pip-pkg/* /app/pip-pkg/
COPY asr_api_server-$version-cp38-cp38-linux_x86_64.whl /app
COPY libsndfile/libsndfile.so.1.0.34 /usr/local/lib/

RUN ln -s /usr/local/lib/libsndfile.so.1.0.34 /usr/local/lib/libsndfile.so.1 \
    && ln -s /usr/local/lib/libsndfile.so.1.0.34 /usr/local/lib/libsndfile.so

# prepaire for build fasttext
RUN echo "deb https://mirrors.nju.edu.cn/debian/ bullseye main contrib non-free"  > /etc/apt/sources.list \
    && apt-get update && apt-get install -y  --no-install-recommends libflac8 libvorbis0a libopus0 libogg0 libmpg123-0 libmp3lame0 libvorbisenc2 g++ \
    && pip install --no-cache-dir pip-pkg/* \
    && pip install --no-cache-dir asr_api_server-$version-cp38-cp38-linux_x86_64.whl \
    && rm -rf pip-pkg \
    && rm *.whl \
    && apt-get remove g++ -y \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/apt/lists/*

CMD ["asr_api_server"]
