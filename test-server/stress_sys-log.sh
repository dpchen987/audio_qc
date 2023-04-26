#!/bin/bash
#
now=$(date +%Y-%m-%d_%H:%M:%S)
while true; do (echo "%CPU %MEM ARGS $(date +%Y-%m-%d_%H:%M:%S)" && ps -e -o pcpu,pmem,args --sort=pcpu | cut -d" " -f1-5 | tail) >> sys-$now.log; sleep 1; done

