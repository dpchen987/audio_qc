FROM python:3.8-slim
WORKDIR /workspace
COPY *.so* run_funasr_websocket funasr-wss-server server.crt server.key ./
CMD ["./run_funasr_websocket"]

