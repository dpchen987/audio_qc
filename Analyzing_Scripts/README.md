# log 分析

- step1: 导出asr-api日志
```sh
docker logs <asr_api_server_name> > api.log
```

- step2: 解析识别时间
    + 1. rec时间：代码内部计算识别时间
    + 2. task时间：根据task_id出现首尾时间作差。

- step3: 统计
    + 见 `show_logs.ipynb`
  
# GPU 使用率分析
```shell
bash test-server/stress_sys-log.sh
```
