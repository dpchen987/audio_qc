from .config import CONF
from .redismq import RedisMQ


mq = RedisMQ(CONF['redis_ip'], CONF['redis_port'], 'asr', 'asr-g0')
