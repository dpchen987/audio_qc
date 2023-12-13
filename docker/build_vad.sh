#!/bin/bash

version=$(python get_version.py)
echo "version: $version"
pyversion=310

asr_api_server-0.9.0-cp310-cp310-linux_x86_64.whl
vad_api_py_whl=asr_api_server-"$version"-py3-none-any.whl
vad_api_so_whl=asr_api_server-"$version"-cp"$pyversion"-cp"$pyversion"-linux_x86_64.whl

if [ -e $vad_api_so_whl ]; then
  vad_api_whl=$vad_api_so_whl
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  echo "use old $vad_api_whl"
  echo 'if want to build new one, please delete the old one'
  echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
else
  cd ..
  rm -rf build dist
  python setup_vad.py bdist_wheel
  find . -print0 -name "*.c" | xargs rm
  if [ -e dist/$vad_api_so_whl ]; then
    vad_api_whl=$vad_api_so_whl
  else
    vad_api_whl=$vad_api_py_whl
  fi
  cp dist/$vad_api_whl docker/
  cd docker
fi


cp ../requirements_vad.txt .

# gen Dockerfile
cat << EOF > Dockerfile_vad
FROM python:3.10-slim
WORKDIR /app
ENV TZ="Asia/Shanghai"
COPY lib /usr/local/lib/
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free"  > /etc/apt/sources.list \
  && echo "deb https://security.debian.org/debian-security bookworm-security main contrib non-free" >> /etc/apt/sources.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends libflac-dev libvorbis0a libopus0 libogg0 libmpg123-0 libmp3lame0 libvorbisenc2 \
  && rm -rf /var/lib/apt/lists/*
COPY soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl requirements_vad.txt /app/
RUN pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple some-package --no-cache-dir soundfile-0.12.1-py2.py3-none-manylinux_2_17_x86_64.whl -r requirements_vad.txt
COPY $vad_api_whl /app/
RUN pip install --no-cache-dir $vad_api_whl && rm $vad_api_whl

CMD [ "vad_api_server"]
EOF

# build docker
docker build -f Dockerfile_vad -t vad_api_server:$version .

# save docker image, give it to IT guys for deploying
# docker save -o image-chatcare-$version-`date "+%Y-%m-%d_%H.%M.%S"`.tar vad_api_server:$version

# clear
rm requirements_vad.txt
