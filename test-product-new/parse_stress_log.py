#!/usr/bin/env python3
import json
import re
import time 
from pprint import pprint


def ts2t(ts):
    ts = ts.replace('_', ' ')
    tm = time.strptime(ts, '%Y-%m-%d %H:%M:%S')
    t = time.mktime(tm)
    return t 


def export_text(client_log, callback_log):
    with open(client_log) as f:
        taskids = json.load(f)
    texts = []
    with open(callback_log) as f:
        for line in f:
            taskid = re.findall(r"{'task_id': '(.*?)'", line)
            if not taskid:
                continue
            taskid = taskid[0]
            if taskid not in taskids:
                continue
            fname = taskid.split('--')[1].split('-')[0]
            content = re.findall(r"'content': '(.*?)'", line)[0]
            line = f'{fname}\t{content}\n'
            texts.append(line)
    print(f'{len(texts) = }')
    with open('sample-2985-texts.txt', 'w') as f:
        f.write(''.join(texts))
    

def parse(client_log, callback_log):
    with open(client_log) as f:
        taskids = json.load(f)
    sent_taskids = set(taskids)
    callback_tasks = []
    with open(callback_log) as f:
        for line in f:
            taskid = re.findall(r"{'task_id': '(.*?)'", line)
            if not taskid:
                continue
            taskid = taskid[0]
            if taskid not in taskids:
                continue
            start_time = taskid.split('--')[0]
            back_time = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            back_time = back_time[0]
            cost = ts2t(back_time) - ts2t(start_time)
            callback_tasks.append({
                'taskid': taskid,
                'start_time': start_time,
                'back_time': back_time,
                'cost_time': cost
            })
    print(f'{len(sent_taskids) = }')
    print(f'{len(callback_tasks) = }')
    start_times = [i['start_time'] for i in callback_tasks]
    start_times.sort()
    back_times = [i['back_time'] for i in callback_tasks]
    back_times.sort()
    print(f'from: {start_times[0]}, to: {back_times[-1]}')
    duration = ts2t(back_times[-1]) - ts2t(start_times[0])
    print(f'{duration = }')
    

if __name__ == "__main__":
    from sys import argv
    client_log = argv[1]
    callback_log = argv[2]
    export_text(client_log, callback_log)
    parse(client_log, callback_log)



