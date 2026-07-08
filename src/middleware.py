"""安全中间件 — 速率限制 + 输入校验 + 请求追踪。"""

import hashlib
import time
import threading
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════
# 速率限制器（滑动窗口 + 令牌桶混合）
# ═══════════════════════════════════════════════════════════════

class RateLimiter:
    """基于滑动窗口的速率限制器。

    支持按 IP 和按 endpoint 两个维度的限制。
    线程安全。
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._storage: Dict[str, list] = {}  # key → [timestamps]
        self._lock = threading.Lock()
        self._cleanup_interval = 300  # 5 分钟清理一次过期数据
        self._last_cleanup = time.time()

    def _cleanup(self):
        """清理过期的请求记录。"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        cutoff = now - self.window_seconds
        with self._lock:
            expired_keys = []
            for key, timestamps in self._storage.items():
                self._storage[key] = [t for t in timestamps if t > cutoff]
                if not self._storage[key]:
                    expired_keys.append(key)
            for key in expired_keys:
                del self._storage[key]
        self._last_cleanup = now

    def check(self, key: str, cost: int = 1) -> Tuple[bool, int]:
        """检查是否允许请求。

        Returns:
            (allowed, remaining) — 是否允许 + 剩余配额
        """
        self._cleanup()
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            if key not in self._storage:
                self._storage[key] = []

            # 清理过期
            self._storage[key] = [t for t in self._storage[key] if t > cutoff]

            current = len(self._storage[key])
            if current + cost > self.max_requests:
                return False, max(0, self.max_requests - current)

            self._storage[key].extend([now] * cost)
            return True, self.max_requests - current - cost

    def reset(self, key: str = None):
        """重置计数器。key 为 None 时重置全部。"""
        with self._lock:
            if key is None:
                self._storage.clear()
            elif key in self._storage:
                del self._storage[key]


# 全局单例
limiter = RateLimiter(max_requests=60, window_seconds=60)


# ═══════════════════════════════════════════════════════════════
# 输入校验
# ═══════════════════════════════════════════════════════════════

MAX_QUESTION_LENGTH = 200_000  # 问题最大字符数（含附件）
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
BANNED_PATTERNS = [
    r"<script",           # XSS
    r"javascript:",        # XSS
    r"on\w+\s*=",         # inline event handlers
    r"\.\./\.\./",        # path traversal
]


def validate_question(text: str) -> Tuple[bool, str]:
    """校验用户输入的安全性。

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "问题不能为空"

    if len(text) > MAX_QUESTION_LENGTH:
        return False, f"问题过长（最大 {MAX_QUESTION_LENGTH} 字符）"

    for pattern in BANNED_PATTERNS:
        import re
        if re.search(pattern, text, re.IGNORECASE):
            return False, "输入包含不安全内容"

    return True, ""


# ═══════════════════════════════════════════════════════════════
# 请求追踪
# ═══════════════════════════════════════════════════════════════

class RequestTracker:
    """请求计数和延迟统计。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_requests = 0
        self.active_requests = 0
        self.total_errors = 0
        self._latencies: list = []  # 最近 1000 次延迟

    def start_request(self):
        with self._lock:
            self.active_requests += 1
            self.total_requests += 1
        return time.time()

    def end_request(self, start_time: float, is_error: bool = False):
        latency = time.time() - start_time
        with self._lock:
            self.active_requests -= 1
            if is_error:
                self.total_errors += 1
            self._latencies.append(latency)
            if len(self._latencies) > 1000:
                self._latencies = self._latencies[-1000:]

    def stats(self) -> dict:
        with self._lock:
            lats = self._latencies[:]
        if not lats:
            return {
                "total_requests": self.total_requests,
                "active_requests": self.active_requests,
                "total_errors": self.total_errors,
            }
        lats.sort()
        return {
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "total_errors": self.total_errors,
            "latency_p50": lats[len(lats) // 2],
            "latency_p95": lats[int(len(lats) * 0.95)],
            "latency_p99": lats[int(len(lats) * 0.99)],
        }


tracker = RequestTracker()


# ═══════════════════════════════════════════════════════════════
# Django 中间件适配
# ═══════════════════════════════════════════════════════════════

def get_client_ip(request) -> str:
    """获取客户端 IP（支持代理）。"""
    x_forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if x_forwarded:
        return x_forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "unknown")


def check_rate_limit(request) -> Tuple[bool, str]:
    """Django 视图用的速率限制检查。"""
    ip = get_client_ip(request)
    endpoint = request.path
    key = f"{ip}:{endpoint}"

    allowed, remaining = limiter.check(key)
    if not allowed:
        return False, f"请求过于频繁，请 {limiter.window_seconds} 秒后再试。剩余配额: {remaining}"

    # 同时检查全局 IP 限制
    allowed_ip, _ = limiter.check(ip)
    if not allowed_ip:
        return False, "请求过于频繁，请稍后再试。"

    return True, ""


# ═══════════════════════════════════════════════════════════════
# Token 消耗监控 + 成本统计
# ═══════════════════════════════════════════════════════════════

class TokenTracker:
    """Token 消耗和成本统计（按模块）。DeepSeek-chat: 输入 ¥1/百万token, 输出 ¥2/百万token。"""
    _COST_PER_1K_INPUT = 0.001
    _COST_PER_1K_OUTPUT = 0.002

    def __init__(self):
        self._lock = __import__('threading').Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self._hourly = []
        self._by_module = {}  # {module: {input, output, count}}

    def record(self, input_tokens: int, output_tokens: int, module: str = "", user: str = ""):
        import time
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_requests += 1
            self._hourly.append((time.time(), input_tokens + output_tokens))
            if module:
                m = self._by_module.get(module, {"input": 0, "output": 0, "count": 0, "cost": 0.0})
                m["input"] += input_tokens
                m["output"] += output_tokens
                m["count"] += 1
                m["cost"] += (input_tokens / 1000 * self._COST_PER_1K_INPUT + output_tokens / 1000 * self._COST_PER_1K_OUTPUT)
                self._by_module[module] = m
            cutoff = time.time() - 3600
            self._hourly = [(t, n) for t, n in self._hourly if t > cutoff]
            # 按模块
            if module:
                m = self._by_module.get(module, {"input": 0, "output": 0, "count": 0})
                m["input"] += input_tokens
                m["output"] += output_tokens
                m["count"] += 1
                self._by_module[module] = m
            # 按用户
            if user:
                u = self._by_module.get("user:" + user, {"input": 0, "output": 0, "count": 0})
                u["input"] += input_tokens
                u["output"] += output_tokens
                u["count"] += 1
                self._by_module["user:" + user] = u

    @property
    def total_cost(self) -> float:
        return (self.total_input_tokens / 1000 * self._COST_PER_1K_INPUT +
                self.total_output_tokens / 1000 * self._COST_PER_1K_OUTPUT)

    def stats(self) -> dict:
        with self._lock:
            by_mod = dict(self._by_module)
        total = self.total_input_tokens + self.total_output_tokens
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total,
            "total_cost_yuan": round(self.total_cost, 4),
            "hourly_tokens": sum(n for _, n in self._hourly),
            "by_module": {k: {"input": v["input"], "output": v["output"], "count": v["count"],
                               "cost_yuan": round((v["input"]/1000*self._COST_PER_1K_INPUT + v["output"]/1000*self._COST_PER_1K_OUTPUT), 4)}
                          for k, v in sorted(by_mod.items(), key=lambda x: x[1]["input"]+x[1]["output"], reverse=True)},
        }


    def stats_by_module(self) -> dict:
        with self._lock:
            return {"total_input": self.total_input_tokens, "total_output": self.total_output_tokens,
                    "total_cost": round(self.total_cost, 4), "total_requests": self.total_requests,
                    "by_module": self._by_module, "hourly_tokens": sum(n for _, n in self._hourly)}


token_tracker = TokenTracker()


# ═══════════════════════════════════════════════════════════════
# Prompt 注入防护
# ═══════════════════════════════════════════════════════════════

import re as _re

def check_prompt_injection(text: str):
    patterns = [
        r"(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|commands?)",
        r"(?:you\s+(?:are|now)\s+)(?:a\s+)?(?:DAN|jailbreak|evil|unethical|unrestricted)",
        r"system\s*(?:prompt|message|instruction)\s*(?:is|was|has been|:)",
        r"<\s*(?:script|iframe|img|svg)",
    ]
    for p in patterns:
        m = _re.search(p, text, _re.IGNORECASE)
        if m:
            return False, f"检测到 Prompt 注入模式: {m.group()[:80]}"
    return True, ""


# ═══════════════════════════════════════════════════════════════
# 数据脱敏
# ═══════════════════════════════════════════════════════════════

def desensitize(text: str) -> str:
    text = _re.sub(r'1[3-9]\d{9}', lambda m: m.group()[:3] + '****' + m.group()[-4:], text)
    text = _re.sub(r'\d{17}[\dXx]', lambda m: m.group()[:6] + '********' + m.group()[-4:], text)
    text = _re.sub(r'(\d+\.?\d*)\s*(?:万元|亿元)', '[金额]', text)
    text = _re.sub(r'(?:密码|password)\s*[：:=]\s*\S+', lambda m: m.group().split(":")[0].split("：")[0] + ': ***', text)
    return text


# ═══════════════════════════════════════════════════════════════
# 审计日志
# ═══════════════════════════════════════════════════════════════

class AuditLogger:
    def __init__(self, max_entries: int = 2000):
        self._lock = __import__('threading').Lock()
        self._entries = []
        self._max = max_entries

    def log(self, user: str, action: str, detail: str = "", status: str = "ok"):
        import time
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user": user or "anonymous",
            "action": action,
            "detail": detail[:200],
            "status": status,
        }
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]

    def list(self, limit: int = 100) -> list:
        with self._lock:
            return list(reversed(self._entries[-limit:]))

    def stats(self) -> dict:
        with self._lock:
            if not self._entries:
                return {"total": 0}
            users = set(e["user"] for e in self._entries)
            acts = {}
            for e in self._entries:
                acts[e["action"]] = acts.get(e["action"], 0) + 1
            return {"total": len(self._entries), "unique_users": len(users), "by_action": acts}


audit_log = AuditLogger()
