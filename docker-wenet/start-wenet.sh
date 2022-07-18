#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
export GLOG_logtostderr=${GLOG_logtostderr:=1}
export GLOG_v=${GLOG_v:=2}
SDC_WS_PORT=${SDC_WS_PORT:=8301}
SDC_WS_CHUNK_SIZE=${SDC_WS_CHUNK_SIZE:=-1}
SDC_WS_CONTEXT_SCORE=${SDC_WS_CONTEXT_SCORE:=8}
echo "GLOG_logtostderr: $GLOG_logtostderr"
echo "GLOG_v: $GLOG_v"
echo "SDC_WS_PORT: $SDC_WS_PORT"
echo "SDC_WS_CHUNK_SIZE: $SDC_WS_CHUNK_SIZE"
echo "SDC_WS_CONTEXT_SCORE: $SDC_WS_CONTEXT_SCORE"
model_dir=./models
./websocket_server_main \
    --port $SDC_WS_PORT \
    --chunk_size $SDC_WS_CHUNK_SIZE \
    --context_score $SDC_WS_CONTEXT_SCORE \
    --context_path $model_dir/hotext.txt \
    --model_path $model_dir/final.zip \
    --unit_path $model_dir/words.txt 2>&1 | tee server.log
