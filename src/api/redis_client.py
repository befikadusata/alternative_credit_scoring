import json
import logging
import os
from threading import Lock
from typing import Optional

import redis

from src.api.models import PredictionInput

logger = logging.getLogger(__name__)


class RedisClient:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance - used for testing."""
        with cls._lock:
            cls._instance = None

    def _initialize(self):
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        self.REDIS_DB = int(os.getenv("REDIS_DB", 0))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
        self.REDIS_TTL_SECONDS = int(
            os.getenv("REDIS_TTL_SECONDS", 3600)
        )  # Default to 1 hour

        self.client: Optional[redis.Redis] = None
        self._connect()

    def _connect(self):
        try:
            self.client = redis.Redis(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                db=self.REDIS_DB,
                password=self.REDIS_PASSWORD,
                socket_connect_timeout=1,
            )
            self.client.ping()
            logger.info("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"An unexpected error occurred while connecting to Redis: {e}")
            self.client = None

    def _get_redis_key(self, loan_id: str) -> str:
        return f"loan_id:{loan_id}"

    def get_features(self, loan_id: str) -> Optional[PredictionInput]:
        if not self.client:
            logger.warning("Redis client not connected. Cannot get features.")
            return None

        try:
            key = self._get_redis_key(loan_id)
            cached_data = self.client.get(key)
            if cached_data:
                logger.info(f"Features for loan_id '{loan_id}' found in Redis cache.")
                # Assuming the stored data is a JSON string representation of the features
                features_dict = json.loads(cached_data)
                return PredictionInput(**features_dict)
            logger.info(f"Features for loan_id '{loan_id}' not found in Redis cache.")
            return None
        except Exception as e:
            logger.error(
                f"Error retrieving features for loan_id '{loan_id}' from Redis: {e}"
            )
            return None

    def set_features(self, loan_id: str, features: PredictionInput) -> bool:
        if not self.client:
            logger.warning("Redis client not connected. Cannot set features.")
            return False

        try:
            key = self._get_redis_key(loan_id)
            # Store the Pydantic model as a JSON string
            self.client.setex(key, self.REDIS_TTL_SECONDS, features.json())
            logger.info(
                f"Features for loan_id '{loan_id}' stored in Redis cache with TTL {self.REDIS_TTL_SECONDS}s."
            )
            return True
        except Exception as e:
            logger.error(
                f"Error storing features for loan_id '{loan_id}' to Redis: {e}"
            )
            return False


def get_redis_client():
    """Get the Redis client instance (singleton)."""
    return RedisClient()

# Ensure the client is initialized as a module-level instance (singleton)
# This maintains backward compatibility with existing imports
redis_client = RedisClient()
