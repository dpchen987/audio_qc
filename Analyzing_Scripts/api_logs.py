import re
import pandas as pd


def parse_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    rec_list = []
    task_list_enter = []
    task_list_complete = []
    for data in lines:
        pattern_rec = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*time use:(\d+\.\d+)'
        durations_rec = re.findall(pattern_rec, data)
        if durations_rec:
            rec_list.append(durations_rec[0])

        # task_id
        pattern_task_enter = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*task: (\d+) enter api'
        pattern_task_complete = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*task_id\': \'(\d+)\', \'code':"
        durations_task_enter = re.findall(pattern_task_enter, data)
        durations_task_complete = re.findall(pattern_task_complete, data)
        if durations_task_enter:
            task_list_enter.append(durations_task_enter[0])
        if durations_task_complete:
            task_list_complete.append(durations_task_complete[0])

    df_rec = pd.DataFrame(rec_list, columns=['time', 'rec'])
    df_rec['time'] = pd.to_datetime(df_rec['time'])
    df_rec['rec'] = df_rec['rec'].astype(float)
    df_rec.drop_duplicates(inplace=True)
    df_rec.sort_values(by='time', inplace=True)

    task_enter_dict = {task_id: enter_time for enter_time, task_id in task_list_enter}
    task_list = []
    for complete_time, task_id in task_list_complete:
        if task_id in task_enter_dict:
            task_list.append((task_id, task_enter_dict[task_id], complete_time))

    df_task = pd.DataFrame(task_list, columns=['task_id', 'enter_time', 'complete_time'])
    df_task['enter_time'] = pd.to_datetime(df_task['enter_time'])
    df_task['complete_time'] = pd.to_datetime(df_task['complete_time'])
    df_task['time'] = df_task['enter_time']
    df_task.drop_duplicates(inplace=True)
    df_task.sort_values(by='task_id', inplace=True)
    return df_rec, df_task


if __name__ == '__main__':
    log_file = r'/aidata/junjie/data/yj-anls/task_asr_evaluate/docker_logs_anls/16/demo.log'
    df_rec, df_task = parse_log(log_file)
    print(df_rec)
    print(df_task)
