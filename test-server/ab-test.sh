ab -n 20 -c 10 -p ab_post.txt \
	-H "accept: application/json" \
	-H "appkey: 123" \
	-H "format: pcm" \
	-H "audio-url: http://127.0.0.1:8010/z-60s.wav" \
	http://127.0.0.1:8300/asr/v1/rec
