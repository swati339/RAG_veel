# import redis
# import os
# import json
# import logging
# from redis.exceptions import RedisError
# from rag_chat.configs.logging_config import setup_logging

# # Setup logger
# setup_logging()
# logger = logging.getLogger(__name__)

# # Load Redis environment configuration
# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
# REDIS_DB = int(os.getenv("REDIS_DB", 0))


# class RedisSetup:
#     def __init__(self):
#         try:
#             self.redis_client = redis.Redis(
#                 host=REDIS_HOST,
#                 port=REDIS_PORT,
#                 db=REDIS_DB,
#                 decode_responses=True  # Automatically decode byte strings
#             )
#             logger.info("Redis client initialized successfully.")
#         except RedisError as e:
#             logger.error(f"Error initializing Redis client: {e}")
#             raise

#     def redis_hset(self, key: str, field: str, value):
#         """Set a field in a Redis hash."""
#         try:
#             if isinstance(value, (dict, list)):
#                 value = json.dumps(value)
#             elif not isinstance(value, str):
#                 value = str(value)
#             self.redis_client.hset(name=key, key=field, value=value)
#             logger.info(f"HSET: {key}[{field}] set successfully.")
#         except RedisError as e:
#             logger.error(f"Redis HSET error: {e}")

#     def redis_hget(self, key: str, field: str):
#         """Get a specific field from a Redis hash."""
#         try:
#             value = self.redis_client.hget(name=key, key=field)
#             if value is None:
#                 logger.info(f"No hash field found: {key}[{field}]")
#                 return None
#             try:
#                 return json.loads(value)
#             except json.JSONDecodeError:
#                 return value
#         except RedisError as e:
#             logger.error(f"Redis HGET error: {e}")
#             return None

#     def redis_hgetall(self, key: str):
#         """Get all fields and values from a Redis hash."""
#         try:
#             raw_data = self.redis_client.hgetall(name=key)
#             parsed_data = {}
#             for field, value in raw_data.items():
#                 try:
#                     parsed_data[field] = json.loads(value)
#                 except json.JSONDecodeError:
#                     parsed_data[field] = value
#             return parsed_data
#         except RedisError as e:
#             logger.error(f"Redis HGETALL error: {e}")
#             return {}

#     def redis_hdel(self, key: str, field: str):
#         """Delete a specific field from a Redis hash."""
#         try:
#             self.redis_client.hdel(key, field)
#             logger.info(f"Deleted hash field: {key}[{field}]")
#         except RedisError as e:
#             logger.error(f"Redis HDEL error: {e}")

import redis
import os
import json
import logging
from redis.exceptions import RedisError
from app.configs.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

class RedisSetup:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            logger.info("Redis client initialized successfully.")
        except RedisError as e:
            logger.error(f"Error initializing Redis client: {e}")
            raise

    def redis_hset(self, key: str, field: str, value):
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)
            self.redis_client.hset(name=key, key=field, value=value)
            logger.info(f"HSET: {key}[{field}] set successfully.")
        except RedisError as e:
            logger.error(f"Redis HSET error: {e}")

    def redis_hget(self, key: str, field: str):
        try:
            value = self.redis_client.hget(name=key, key=field)
            if value is None:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except RedisError as e:
            logger.error(f"Redis HGET error: {e}")
            return None
