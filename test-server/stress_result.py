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
    all_begin = time.time()
    all_end = 0
    for c in client:
        taskid = c['taskid']
        if taskid not in callback:
            print('no callback result for', taskid)
            failed.add(taskid)
            continue
        if c['begin'] < all_begin:
            all_begin = c['begin']
        if callback[taskid] > all_end:
            all_end = callback[taskid]
        past = callback[taskid] - c['begin']
        task_times.append(past)
    duration = len(task_times) * 60  # 60s per audio
    cost = all_end - all_begin
    print('failed:', len(failed), failed)
    print('RTF:', round(cost/duration, 5))
    print('spped:', round(duration/cost, 3))
    print('mean: ', round(statistics.mean(task_times), 3))
    print('median:', round(statistics.median(task_times), 3))
    print('max_time:', round(max(task_times), 3))
    print('min_time:', round(min(task_times), 3))


if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 3:
        print('Usage:', f'{argv[0]} client-log callback-log')
        exit()
    fn_client = argv[1]
    fn_callback = argv[2]
    parse(fn_client, fn_callback)





