FROM ubuntu:latest

WORKDIR /wenet
COPY models ./models
COPY *.so* websocket_server_main start-wenet.sh ./

CMD ["./start-wenet.sh"]
