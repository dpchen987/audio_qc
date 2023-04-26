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


def parse_sys_log(fn, process=['asr_api_server', 'tritonserver']):
    data = {}  # {time: [%cpu, %mem], }
    dtime = ''
    item = {}
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if not dtime and line[0] != '%':
                continue
            if line[0] == '%':
                if dtime:
                    data[dtime] = item
                dtime = line.split(' ')[-1]
                dtime = time.mktime(time.strptime(dtime, '%Y-%m-%d_%H:%M:%S'))
                item = {}
                continue
            zz = re.split(r'\s+', line)
            pcpu = float(zz[0])
            pmem = float(zz[1])
            item['pcpu'] = item.get('pcpu', 0) + pcpu
            item['pmem'] = item.get('pmem', 0) + pmem
            pp = ''
            for p in process:
                if p in line:
                    pp = p
                    break
            if p:
                item[p] = (pcpu, pmem)
    if item:
        data[dtime] = item
    return data


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
    return all_begin, all_end


if __name__ == '__main__':
    from sys import argv, exit
    if len(argv) != 4:
        print('Usage:', f'{argv[0]} client-log callback-log sys-log')
        exit()
    fn_client = argv[1]
    fn_callback = argv[2]
    fn_syslog = argv[3]
    test_begin, test_end = parse(fn_client, fn_callback)
    print(f'{test_begin = }, {test_end = }')
    data = parse_sys_log(fn_syslog)
    from pprint import pprint
    pprint(data)





