# asr-server

本项目是给 业务调用ASR服务的，其功能包括：

1. 权限验证
2. VAD
3. 访问 wenet websocket 服务进行ASR

## 运行

首先要先运行wenet websocket服务，测试服务器有可执行环境可测试： /mnt/ai/share/wenet-demo/docker
执行 `run-in-host.sh` 即可，执行前可修改其中的port。

然后再运行asr-server，如下：

```bash
git clone -b develop http://dev.day-care.cn/gitlab/yshy/asr-server.git
cd asr-server
python -m sdc_asr_server.main
```


## 注意事项

本项目使用async 定义 fastapi 接口函数，但是通过Cython编译成 .so 文件后会报错，这应该是 Cython 的问题。
work around 的方法是修改fastapi的源码文件 routing.py 里面的 get_request_handler() ：

```python
def get_request_handler(

    . . . . . .

) -> Callable:
    assert dependant.call is not None, "dependant.call must be a function"
    is_coroutine = asyncio.iscoroutinefunction(dependant.call)
    is_body_form = body_field and isinstance(body_field.field_info, params.Form)

    async def app(request: Request) -> Response:

        . . . . . . 

        if errors:
            raise RequestValidationError(errors, body=body)
        else:
            raw_response = await run_endpoint_function(
                dependant=dependant, values=values, is_coroutine=is_coroutine
            )

            ####### Insert here #######
            if asyncio.iscoroutine(raw_response):
                raw_response = await raw_response
            ###########################

            if isinstance(raw_response, Response):
                if raw_response.background is None:
                    raw_response.background = background_tasks
                return raw_response

            . . . . . .

            return response

    return app
```
