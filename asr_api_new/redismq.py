import traceback
import asyncio
import redis.asyncio as redis


class RedisMQ:
    def __init__(self, host, port, stream_name, group_name):
        self.stream_name = stream_name
        self.group_name = group_name
        self.url = f"redis://{host}:{port}"
        self.conn = None

    async def init(self):
        self.conn = await redis.from_url(self.url)
        try:
            await self.conn.xgroup_create(
                name=self.stream_name,
                groupname=self.group_name, id=0,
                mkstream=True,
            )
        except Exception as e:
            print(e)
        res = await self.conn.xinfo_groups(name=self.stream_name)
        skey = self.stream_name
        has_group = False
        for i in res:
            print(
                (f"{skey} -> group name: {i['name']} with "
                    f"{i['consumers']} consumers and {i['last-delivered-id']}"
                    f" as last read id")
            )
            if i['name'].decode() == self.group_name:
                has_group = True
        if not has_group:
            await self.conn.xgroup_create(
                name=self.stream_name,
                groupname=self.group_name, id=0,
                mkstream=True,
            )

    async def put(self, message):
        await self.conn.xadd(self.stream_name, message)

    async def pop(self, count, consumername="c", block=500):
        try:
            response = await self.conn.xreadgroup(
                groupname=self.group_name,
                consumername=consumername,
                streams={self.stream_name: '>'},
                count=count,
                block=block
            )
            # response: [[stream_name, messages]]
            if response:
                return response[0][1]
        except asyncio.CancelledError:
            print("Consumer is shutting down.")
        except Exception as e:
            traceback.print_exc()
            print(f"An error occurred: {e}")
        return []

    async def claim(self, count, consumername, min_idle_time, start_id='0-0'):
        res = await self.conn.xautoclaim(
            self.stream_name,
            self.group_name,
            consumername,
            min_idle_time,
            start_id=start_id,
            count=count,
        )
        if res:
            return res[1]
        return []

    async def ack(self, message_id):
        return await self.conn.xack(
            self.stream_name, self.group_name, message_id)

    async def pending(self, count, consumername='c'):
        return await self.conn.xpending_range(
            self.stream_name, self.group_name,
            '-', '+', count, consumername=consumername)
