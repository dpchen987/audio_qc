FROM ubuntu:latest

WORKDIR /wenet
COPY wenet-bin/* ./
COPY models ./models
COPY start-wenet.sh ./

CMD ["./start-wenet.sh"]
