nohup uvicorn asr_api_server.main:app --host 0.0.0.0 --port 8002  --workers 1  &> running_log.out &