# build docker, update version if needs
bin_dir=$1
if [ -z $bin_dir ]; then
    echo "please input the dir of wenet bin"
    exit -1
fi
cp Dockerfile $bin_dir
cp start-wenet.sh $bin_dir
cd $bin_dir
version=0.1.0
name=asr_decode_server
docker build -t $name:$version .

# save docker image, give it to IT guys for deploying
# docker save -o image-$name-$version-`date "+%Y-%m-%d_%H:%M:%S"`.tar name_server:$version
