ab -n 150 -c 50 -p ab_post.txt \
	-H "accept: application/json" \
	-H "appkey: 123" \
	-H "format: pcm" \
	-H "audio-url: https://ahc-audio.oss-cn-shanghai-internal.aliyuncs.com/video/SHC0011911260005/1759040041446465536/1629860409787.wav" \
	http://127.0.0.1:8300/asr/v1/rec
