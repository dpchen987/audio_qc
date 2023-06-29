FROM python:3.8-slim
ARG version
WORKDIR /app
ENV TZ="Asia/Shanghai"

# install basic packages
COPY bin requirements.txt leveldb-0.201-cp38-cp38-linux_x86_64.whl soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl /app/
RUN pip install --no-cache-dir leveldb-0.201-cp38-cp38-linux_x86_64.whl soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl -r requirements.txt -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
    && rm requirements.txt leveldb-0.201-cp38-cp38-linux_x86_64.whl soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl

# install latest libsndfile to support opus
COPY lib /usr/local/lib/
RUN ln -s /usr/local/lib/libsndfile.so.1.0.34 /usr/local/lib/libsndfile.so.1 \
    && ln -s /usr/local/lib/libsndfile.so.1.0.34 /usr/local/lib/libsndfile.so

# install libsndfile dependence
RUN echo "deb https://mirrors.bfsu.edu.cn/debian/ bullseye main contrib non-free"  > /etc/apt/sources.list \
    && apt-get update && apt-get install -y  redis-server --no-install-recommends libflac8 libvorbis0a libopus0 libogg0 libmpg123-0 libmp3lame0 libvorbisenc2 \
    && rm -rf /var/lib/apt/lists/*

# install api server packages, change frequently
COPY asr_api_server-$version-cp38-cp38-linux_x86_64.whl /app
RUN pip install --no-cache-dir asr_api_server-$version-cp38-cp38-linux_x86_64.whl && rm *.whl

CMD ["asr_api_server"]
