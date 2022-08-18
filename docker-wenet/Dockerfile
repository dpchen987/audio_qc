FROM ubuntu:latest

WORKDIR /wenet
COPY models ./models
COPY *.so* websocket_server_main run-decoder ./

CMD ["./run-decoder"]
