#!/usr/bin/env python

import json
import time
import re
import statistics


def read_callback(fn_callback):
    result = {}
    with open(fn_callback) as f:
        for line in f:
            m = re.findall(r"'task_id': '(.*?)'", line)
            if not m:
                continue
            task_id = m[0]
            ts = line.split(' - ')[0]
            ts, ms = ts.split(',')
            tm = time.strptime(ts, '%Y-%m-%d %H:%M:%S')
            t = time.mktime(tm)
            result[task_id] = t + int(ms)/1000
    return result


def parse(fn_client, fn_callback):
    with open(fn_client) as f:
        client = json.load(f)
    callback = read_callback(fn_callback)
    task_times = []
    failed = set()
    for c in client:
        taskid = c['taskid']
        if taskid not in callback:
            failed.add(taskid)
            continue
        past = callback[taskid] - c['begin']
        task_times.append(past)
    print('mean: ', statistics.mean(task_times))
    print('median:', statistics.median(task_times))
    print('max_time:', max(task_times))
    print('min_time:', min(task_times))


if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 3:
        print('Usage:', f'{argv[0]} client-log callback-log')
        exit()
    fn_client = argv[1]
    fn_callback = argv[2]
    parse(fn_client, fn_callback)





