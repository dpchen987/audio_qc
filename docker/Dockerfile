FROM python:3.8-slim
ARG version
WORKDIR /app
COPY pip-pkg/* /app/pip-pkg/
COPY asr_api_server-$version-cp38-cp38-linux_x86_64.whl /app

# prepaire for build fasttext
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free"  > /etc/apt/sources.list \
    && apt-get update && apt-get install -y libsndfile1 \
    && pip install pip-pkg/* \
    && pip install asr_api_server-$version-cp38-cp38-linux_x86_64.whl \
    && rm -rf pip-pkg \
    && rm *.whl \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/apt/lists/*

CMD [ "asr_api_server"]
