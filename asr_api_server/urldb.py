import leveldb

from .config import CONF


url_db = leveldb.LevelDB(CONF['url_db'])
