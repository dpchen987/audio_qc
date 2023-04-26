import re
import os
import pandas as pd


def parse_triton_metrics(metrics_logs: str) -> pd.DataFrame:
    """
    获取GPU下列信息：
        gpu_uuid
        nv_energy_consumption
        nv_gpu_utilization
        nv_gpu_memory_total_bytes
        nv_gpu_memory_used_bytes
        nv_gpu_power_usage
        nv_gpu_power_limit
    Args:
        metrics_logs (str): 日志信息
    Returns:
        df (pd.DataFrame): gpu日志数据
    """
    gpu_info_fields = ['nv_energy_consumption', 'nv_gpu_utilization', 'nv_gpu_memory_total_bytes',
                       'nv_gpu_memory_used_bytes', 'nv_gpu_power_usage', 'nv_gpu_power_limit']
    df = pd.DataFrame(columns=gpu_info_fields)
    idx = 0
    for field in gpu_info_fields:
        matches = re.findall(field + r'\{gpu_uuid="([^"]+)"\} ([\d.]+)', metrics_logs)
        for match in matches:
            gpu_uuid, val = match[0], match[1]
            df.loc[gpu_uuid, [field]] = [val]
            idx += 1
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'gpu_uuid'}, inplace=True)
    return df


def parse(gpu_logs_dir: str):
    """
    Args:
        gpu_logs_dir (str): gpu日志目录
    """
    df = pd.DataFrame()
    for filename in os.listdir(gpu_logs_dir):
        with open(os.path.join(gpu_logs_dir, filename)) as f:
            metrics_logs = f.read()
        df_logs = parse_triton_metrics(metrics_logs)
        df_logs['time'] = pd.to_datetime(filename.split('.')[0], format="%Y-%m-%d_%H-%M-%S")
        df = pd.concat([df, df_logs])

    df.sort_values(by=['time', 'gpu_uuid'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    file = 'gpu_parse_logs.txt'
    df.to_csv(file)
    print(f"Save gpu logs parse file to --> {file}")


if __name__ == '__main__':
    from sys import argv, exit

    if len(argv) != 2:
        print('Usage:', f'{argv[0]} gpu_logs_dir')
        exit()
    gpu_logs_dir = argv[1]
    parse(gpu_logs_dir)
