# utils/redis_manager.py
import redis
import redis.sentinel
from typing import Optional, Dict, Any
import logging
import streamlit as st


class RedisManager:
    """Enhanced Redis connection manager"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client: Optional[redis.Redis] = None
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> redis.Redis:
        """Establish Redis connection with error handling"""
        try:
            # Create connection pool for better performance
            self.connection_pool = redis.ConnectionPool(
                host=self.config["redis_host"],
                port=self.config["redis_port"],
                db=self.config["redis_db"],
                password=self.config.get("redis_password", None),
                socket_timeout=self.config.get("socket_timeout", 5),
                socket_connect_timeout=self.config.get("socket_connect_timeout", 5),
                max_connections=self.config.get("max_connections", 20),
                decode_responses=True,
                ssl=self.config.get("ssl", False)
            )

            # Create Redis client
            self.client = redis.Redis(connection_pool=self.connection_pool)

            # Test connection
            self.client.ping()
            self.logger.info("✅ Redis connection established successfully")

            return self.client

        except redis.ConnectionError as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected Redis error: {e}")
            return None

    def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client with auto-reconnection"""
        if not self.client:
            return self.connect()

        try:
            # Test if connection is alive
            self.client.ping()
            return self.client
        except:
            # Reconnect if connection is dead
            self.logger.warning("🔄 Redis connection lost, reconnecting...")
            return self.connect()

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
        if self.connection_pool:
            self.connection_pool.disconnect()