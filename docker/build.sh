#!/bin/bash
# get app_version

app_version=$(python get_app_version.py)
echo "app_version: $app_version"
py_version=$(python get_pyversion.py)
echo "py_version: $py_version"

app_whl=asr_api_server-"$app_version"-cp"$py_version"-cp"$py_version"-linux_x86_64.whl

# build asr_api_server.*.whl if not exists here
if [ -e $app_whl ]; then
    echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    echo "use old $app_whl"
    echo 'if want to build new one, please delete the old one'
    echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
else
    cd ..
    rm -rf build dist
    python setup.py bdist_wheel
    # cd asr_api_server && find . -name "*.c" | xargs rm
    # cd ..
    cp dist/$app_whl docker/
    cd docker
fi

# prepare pip wheel to avoid building in Docker
if ! ls uvloop*-cp$py_version*.whl >/dev/null 2>&1; then
  pip wheel uvloop
fi 

if ! ls numpy*-cp$py_version*.whl >/dev/null 2>&1; then
  pip wheel numpy
fi 

if ! ls scikit_learn*-cp$py_version*.whl >/dev/null 2>&1; then
  pip wheel scikit_learn
fi 

# gen Dockerfile
cat << EOF > Dockerfile_asr
FROM python:3.10-slim-bullseye
WORKDIR /app
ENV TZ="Asia/Shanghai"

# install libsndfile dependence
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bullseye main contrib non-free"  > /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends libflac8 libvorbis0a libopus0 libogg0 libmpg123-0 libmp3lame0 libvorbisenc2 \
    && rm -rf /var/lib/apt/lists/*

# install basic wheel
COPY scikit_learn*$py_version*.whl \
    joblib*.whl \
    scipy*cp$py_version*.whl \
    threadpoolctl*.whl \
    uvloop*cp$py_version*.whl \
    numpy-1.26.3-cp$py_version-cp$py_version-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    leveldb-0.201-cp$py_version-cp$py_version-linux_x86_64.whl \
    soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl \
    /app/
RUN pip install --no-cache-dir *.whl -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && rm *.whl

COPY requirements.txt /app/
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && rm requirements.txt

# install api server packages, change frequently
COPY $app_whl /app
RUN pip install --no-cache-dir $app_whl && rm *.whl

CMD ["asr_api_server"]
EOF

docker build -f Dockerfile_asr -t asr_api_server:"$app_version" .


# save docker image, give it to IT guys for deploying
# docker save -o image-asr_api_server-$app_version-`date "+%Y-%m-%d_%H-%M-%S"`.tar asr_api_server:$app_version

