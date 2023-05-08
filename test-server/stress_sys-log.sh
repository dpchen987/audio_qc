#!/bin/bash

now=$(date +%Y-%m-%d_%H:%M:%S)
while true; do

  # CPU MEM monitor
  (echo "%CPU %MEM ARGS $(date +%Y-%m-%d_%H:%M:%S)" && ps -e -o pcpu,pmem,args --sort=pcpu | cut -d" " -f1-5 | tail) >> sys-$now.log

  # GPU monitor
  if [ ! -f sys-gpu-$now.log ]; then
    echo "timestamp, name, uuid, index, pstate, temperature.gpu [C], power.draw [W], power.limit [W], utilization.gpu [%], utilization.memory [%], memory.used [MiB], memory.total [MiB]" > sys-gpu-$now.log
  else
    nvidia-smi --query-gpu=timestamp,gpu_name,gpu_uuid,index,pstate,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv | tail -n +2 | tee -a sys-gpu-$now.log
  fi

  sleep 1
done
