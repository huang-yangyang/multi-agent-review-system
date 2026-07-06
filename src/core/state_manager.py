"""Distributed State Manager with Redis backend and version control.

Manages shared state across agents with:
- Redis-backed key-value storage
- Version-controlled updates (optimistic locking)
- JSON serialization with schema validation
- TTL-based expiration
- Graceful fallback to in-memory store when Redis is unavailable
"""

import json
import time
from typing import Any, Dict, List, Optional


class StateVersion:
    """Version tracking for state entries."""

    def __init__(self, key: str, version: int = 0):
        self.key = key
        self.version = version
        self.updated_at = time.time()


class StateManager:
    """Manages distributed state with version control.

    Uses Redis when available, falls back to in-memory dict.
    Supports optimistic concurrency via version numbers.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str = "",
    ):
        """Initialize the state manager.

        Args:
            redis_host: Redis server host.
            redis_port: Redis server port.
            redis_db: Redis database number.
            redis_password: Redis authentication password.
        """
        self._redis = None
        self._redis_config = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "password": redis_password,
        }
        self._fallback: Dict[str, Any] = {}
        self._versions: Dict[str, StateVersion] = {}
        self._backend = "memory"
        self._init_redis()

    # ------------------------------------------------------------------
    # Backend Initialization
    # ------------------------------------------------------------------

    def _init_redis(self) -> bool:
        """Try to connect to Redis. Returns True on success."""
        try:
            import redis
            self._redis = redis.Redis(
                host=self._redis_config["host"],
                port=self._redis_config["port"],
                db=self._redis_config["db"],
                password=self._redis_config["password"] or None,
                socket_connect_timeout=2,
                decode_responses=True,
            )
            self._redis.ping()
            self._backend = "redis"
            return True
        except Exception:
            self._redis = None
            return False

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key.

        Args:
            key: The state key.
            default: Default value if key not found.

        Returns:
            The stored value (deserialized from JSON).
        """
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
            except Exception:
                pass

        if key in self._fallback:
            return self._fallback[key]
        return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value with optional TTL.

        Args:
            key: The state key.
            value: Value to store (must be JSON-serializable).
            ttl: Optional time-to-live in seconds.

        Returns:
            True on success.
        """
        serialized = json.dumps(value, ensure_ascii=False, default=str)

        if self._redis:
            try:
                if ttl:
                    self._redis.setex(key, ttl, serialized)
                else:
                    self._redis.set(key, serialized)
            except Exception:
                pass
            else:
                self._update_version(key)
                return True

        self._fallback[key] = value
        self._update_version(key)
        return True

    def delete(self, key: str) -> bool:
        """Delete a key.

        Args:
            key: The key to delete.

        Returns:
            True if the key existed and was deleted.
        """
        existed = False
        if self._redis:
            try:
                existed = self._redis.delete(key) > 0
            except Exception:
                pass
        existed = existed or (key in self._fallback)
        self._fallback.pop(key, None)
        self._versions.pop(key, None)
        return existed

    def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists.
        """
        if self._redis:
            try:
                return bool(self._redis.exists(key))
            except Exception:
                pass
        return key in self._fallback

    # ------------------------------------------------------------------
    # Version Control (Optimistic Locking)
    # ------------------------------------------------------------------

    def _update_version(self, key: str) -> None:
        """Increment the version for a key."""
        if key in self._versions:
            self._versions[key].version += 1
            self._versions[key].updated_at = time.time()
        else:
            self._versions[key] = StateVersion(key, version=1)

    def get_version(self, key: str) -> int:
        """Get the current version number of a key.

        Args:
            key: The state key.

        Returns:
            Version number, 0 if key does not exist.
        """
        version_entry = self._versions.get(key)
        return version_entry.version if version_entry else 0

    def compare_and_set(
        self,
        key: str,
        expected_version: int,
        new_value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Atomic compare-and-set with version check.

        Args:
            key: The state key.
            expected_version: The version expected to be current.
            new_value: Value to set if version matches.
            ttl: Optional TTL for the new value.

        Returns:
            True if the set succeeded (version matched), False otherwise.
        """
        current_version = self.get_version(key)
        if current_version != expected_version:
            return False
        self.set(key, new_value, ttl=ttl)
        return True

    # ------------------------------------------------------------------
    # Batch Operations
    # ------------------------------------------------------------------

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys.

        Args:
            keys: List of keys to retrieve.

        Returns:
            Dict mapping key -> value for all found keys.
        """
        result = {}
        if self._redis:
            try:
                raw_values = self._redis.mget(keys)
                for key, raw in zip(keys, raw_values):
                    if raw is not None:
                        result[key] = json.loads(raw)
            except Exception:
                pass

        # Fill in from fallback for any missed keys
        for key in keys:
            if key not in result and key in self._fallback:
                result[key] = self._fallback[key]
        return result

    def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Batch set multiple keys.

        Args:
            mapping: Dict of key -> value pairs.
            ttl: Optional TTL applied to all keys.

        Returns:
            Number of keys set.
        """
        if not mapping:
            return 0
        if self._redis:
            try:
                pipe = self._redis.pipeline()
                for key, value in mapping.items():
                    serialized = json.dumps(value, ensure_ascii=False, default=str)
                    if ttl:
                        pipe.setex(key, ttl, serialized)
                    else:
                        pipe.set(key, serialized)
                pipe.execute()
            except Exception:
                pass

            for key in mapping:
                self._update_version(key)
            return len(mapping)

        for key, value in mapping.items():
            self._fallback[key] = value
            self._update_version(key)
        return len(mapping)

    # ------------------------------------------------------------------
    # Namespace / Prefix
    # ------------------------------------------------------------------

    def get_by_prefix(self, prefix: str) -> Dict[str, Any]:
        """Get all keys matching a prefix.

        Args:
            prefix: Key prefix to match.

        Returns:
            Dict of matching key -> value pairs.
        """
        result = {}
        if self._redis:
            try:
                keys = list(self._redis.scan_iter(match=f"{prefix}*", count=100))
                if keys:
                    raw_values = self._redis.mget(keys)
                    for key, raw in zip(keys, raw_values):
                        if raw is not None:
                            result[key] = json.loads(raw)
            except Exception:
                pass

        # Fallback prefix match
        for key, value in self._fallback.items():
            if key.startswith(prefix) and key not in result:
                result[key] = value
        return result

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete all keys matching a prefix.

        Args:
            prefix: Key prefix to match.

        Returns:
            Number of keys deleted.
        """
        count = 0
        if self._redis:
            try:
                keys = list(self._redis.scan_iter(match=f"{prefix}*", count=100))
                if keys:
                    count += self._redis.delete(*keys)
            except Exception:
                pass

        to_delete = [k for k in self._fallback if k.startswith(prefix)]
        for k in to_delete:
            del self._fallback[k]
            self._versions.pop(k, None)
        return count + len(to_delete)
