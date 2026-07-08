"""系统日志存储 — 环形缓冲区 + Python logging 拦截。"""

import logging
import threading
import time
from typing import Dict, List

MAX_ENTRIES = 500


class LogStore:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self._lock = threading.Lock()
        self._entries: List[Dict] = []
        self._max = max_entries

    def append(self, level: str, module: str, message: str, detail: str = ""):
        with self._lock:
            self._entries.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "level": level, "module": module,
                "message": message[:500], "detail": detail[:1000],
            })
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

    def list(self, limit=200, level="", module=""):
        with self._lock:
            entries = list(self._entries)
        if level:
            entries = [e for e in entries if e["level"] == level.upper()]
        if module:
            entries = [e for e in entries if module.lower() in e["module"].lower()]
        return list(reversed(entries[-limit:]))

    def stats(self):
        with self._lock:
            entries = list(self._entries)
        if not entries:
            return {"total": 0}
        levels, modules = {}, {}
        for e in entries:
            levels[e["level"]] = levels.get(e["level"], 0) + 1
            modules[e["module"]] = modules.get(e["module"], 0) + 1
        return {"total": len(entries), "by_level": levels, "by_module": modules}

    def clear(self):
        with self._lock:
            self._entries.clear()


log_store = LogStore()


class LogStoreHandler(logging.Handler):
    def emit(self, record):
        try:
            module = record.name.split(".")[-1] if record.name != "root" else getattr(record, "module", "system")
            log_store.append(
                level=record.levelname, module=module or "system",
                message=record.getMessage(),
                detail=self.format(record) if record.levelno >= logging.WARNING else "",
            )
        except Exception:
            pass


_handler = LogStoreHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_handler)

# 确保 root logger 级别为 INFO（Django 默认是 WARNING）
logging.getLogger().setLevel(logging.INFO)
