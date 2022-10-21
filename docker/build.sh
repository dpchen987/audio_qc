#!/bin/bash
# get version

version=$(python get_version.py)
echo "version: $version"

# build asr_api_server.*.whl if not exists here
if [ -e asr_api_server-"$version"-cp38-cp38-linux_x86_64.whl ]; then
    echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    echo "use old asr_api_server-$version-cp38-cp38-linux_x86_64.whl"
    echo 'if want to build new one, please delete the old one'
    echo '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
else
    cd ..
    python setup.py bdist_wheel
    find . -print0 -name "*.c" | xargs rm
    cp dist/asr_api_server-"$version"-cp38-cp38-linux_x86_64.whl docker/
    cd docker
fi

# check docker experimental and build
is_experimental=$(docker system info | grep Experimental | grep true)
echo "is_experimental: ($is_experimental)"
if [ -n "$is_experimental" ]; then
    printf "\n============================================================\n"
    echo "Docker run with experimental will get smaller size of image"
    echo "============================================================"
    docker build --build-arg version="$version" --squash -t asr_api_server:"$version" .
else
    printf "\n============================================================\n"
    echo "Docker run without experimental will get bigger size of image"
    echo "To get smaller image, please configure 'experimental' as true"
    echo "============================================================"
    docker build --build-arg version="$version" -t asr_api_server:"$version" .
fi


# save docker image, give it to IT guys for deploying
# docker save -o image-asr_api_server-$version-`date "+%Y-%m-%d_%H-%M-%S"`.tar asr_api_server:$version

# # clear
# rm asr_api_server-$version-cp38-cp38-linux_x86_64.whl
# rm requirements.txt
