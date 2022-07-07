# get version
version=`python get_version.py`
echo "version: $version"

# build asr_api_server.*.whl if not exists here
if [ -e asr_api_server-$version-cp38-cp38-linux_x86_64.whl ]; then
    echo '---------------------------------------'
    echo 'use old asr_api_server-$version-cp38-cp38-linux_x86_64.whl'
    echo 'if want to build new one, please delete the old one'
    echo '---------------------------------------'
else
    cd ..
    python setup.py bdist_wheel
    find . -name *.c | xargs rm
    cp dist/asr_api_server-$version-cp38-cp38-linux_x86_64.whl docker/
    cd docker
fi

# download pip package to 'pip-pkg' if they are not there
if [ -e pip-pkg ]; then
    echo 'has pip-pkg'
else
    mkdir pip-pkg
fi
pip download -r requirements.txt -d pip-pkg

# build docker
docker build --build-arg version=$version -t asr_api_server:$version .


# save docker image, give it to IT guys for deploying
# docker save -o image-asr_api_server-$version-`date "+%Y-%m-%d_%H:%M:%S"`.tar asr_api_server:$version

# # clear
# rm asr_api_server-$version-cp38-cp38-linux_x86_64.whl
# rm requirements.txt
