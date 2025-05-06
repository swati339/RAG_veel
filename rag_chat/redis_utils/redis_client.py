import redis
import os
import json
import logging
from configs.logging_config import setup_logging

# Setup logger
setup_logging()
logger = logging.getLogger(__name__)

# Load Redis environment configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))


class RedisSetup:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True  # Automatically decode byte strings
            )
            logger.info("Redis client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing Redis client: {e}")
            raise

    def redis_set(self, key, value, ex=None):
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)
            self.redis_client.set(key, value, ex=ex)
            logger.info(f"Set key in Redis: {key}")
        except Exception as e:
            logger.error(f"Redis SET error: {e}")

    def redis_get(self, key):
        try:
            value = self.redis_client.get(key)
            if value is None:
                logger.info(f"No cache found for key: {key}")
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value  # Already a string
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None

    def redis_delete(self, key):
        try:
            self.redis_client.delete(key)
            logger.info(f"Deleted key from Redis: {key}")
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
