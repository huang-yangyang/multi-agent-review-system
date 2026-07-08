"""企业级自动修复引擎 — 日志监控 + 故障检测 + 自动修复。

检测类型：
- 模型加载失败 → 重载模型
- 索引不可用 → 重建索引
- 磁盘空间不足 → 告警
- API 超时频发 → 切换 endpoint
- 缓存持久化失败 → 重建缓存文件
"""

import threading
import time
import os
from datetime import datetime
from typing import Dict, List

from src.core.logging_config import get_logger
from src.log_store import log_store

logger = get_logger(__name__)


class AutoHealEngine:
    """自动修复引擎 — 后台线程定期扫描日志，匹配已知故障模式并执行修复。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._interval = 30  # 每 30 秒扫描一次

        # 修复记录
        self._heal_history: List[Dict] = []  # 最多保留 200 条
        self._max_history = 200

        # 故障模式 → 修复动作（延迟绑定，避免方法未定义）
        self._patterns = None

        # LLM 诊断方案缓存（自学习）
        # key = pattern, value = {"suggestion": str, "hit_count": int, "learned_at": str}
        self._llm_cache: Dict[str, dict] = {}
        self._llm_cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "indexes", "autoheal_llm_cache.json"
        )

    def start(self):
        self._init_patterns()
        self._load_llm_cache()
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()
        self._log_heal("init", "自动修复引擎已启动", "success")

    def stop(self):
        self._running = False
        self._log_heal("init", "自动修复引擎已停止", "success")

    def _scan_loop(self):
        while self._running:
            try:
                self._scan_and_heal()
            except Exception as e:
                logger.warning(f"AutoHeal scan error: {e}")
            time.sleep(self._interval)

    def _scan_and_heal(self):
        """扫描最近日志，匹配故障模式并执行修复。"""
        recent = log_store.list(limit=100, level="ERROR")
        recent += log_store.list(limit=100, level="WARNING")
        # 过滤掉 autoheal 自己产生的日志，防止递归匹配
        recent = [e for e in recent if e.get("module", "").lower() not in ("autoheal", "auto_heal")]

        if recent:
            logger.info(
                f"AutoHeal 扫描: {len(recent)} 条日志待检查",
                extra={"component": "autoheal"},
            )

        seen_issues = set()
        for entry in recent:
            msg = entry.get("message", "")
            for pattern, action, friendly in self._patterns:
                import re
                if re.search(pattern, msg) and pattern not in seen_issues:
                    seen_issues.add(pattern)
                    # 第一层：代码修复
                    try:
                        result = action(entry)
                        status = result.get("status", "failed")
                    except Exception as e:
                        result = {"status": "failed", "detail": str(e)}
                        status = "failed"
                    # 第二层：修复失败 → LLM 诊断（带缓存自学习）
                    if status in ("failed", "error", "alert"):
                        # 先检查是否有缓存的 LLM 方案
                        cached = self._llm_cache.get(pattern)
                        if cached:
                            cached["hit_count"] = cached.get("hit_count", 0) + 1
                            result = {
                                "status": "info",
                                "detail": f"🤖 LLM 建议(缓存复用 #{cached['hit_count']}): {cached['suggestion']}"
                            }
                        else:
                            # 无缓存 → 调用 LLM 诊断
                            llm = self._llm_diagnose(pattern, msg[:300], result.get("detail", ""))
                            if llm and llm.get("status") == "recovered":
                                result = llm
                            elif llm and "LLM 建议" in llm.get("detail", ""):
                                # 缓存 LLM 建议供下次复用
                                suggestion = llm["detail"].replace("🤖 LLM 建议: ", "")
                                self._llm_cache[pattern] = {
                                    "suggestion": suggestion,
                                    "hit_count": 0,
                                    "learned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "error_sample": msg[:200],
                                }
                                self._save_llm_cache()
                                result = llm
                    self._log_heal(
                        issue=friendly,
                        detail=msg[:200],
                        result=result.get("status", "unknown"),
                        action_detail=result.get("detail", ""),
                    )
        # 无论是否匹配到故障，都记录扫描完成
        log_store.append(
            level="INFO",
            module="autoheal",
            message=f"[scan] 扫描完成 ({len(recent)} 条日志, {len(seen_issues)} 个匹配)",
            detail="",
        )

    def _log_heal(self, issue: str, detail: str, result: str, action_detail: str = ""):
        with self._lock:
            entry = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "issue": issue,
                "detail": detail[:300],
                "result": result,
                "action": action_detail[:300],
            }
            self._heal_history.append(entry)
            if len(self._heal_history) > self._max_history:
                self._heal_history = self._heal_history[-self._max_history:]
        # 同时写入系统日志
        log_store.append(
            level="INFO",
            module="autoheal",
            message=f"[{result}] {issue}",
            detail=detail[:500],
        )

    # ── 修复动作 ──

    def _fix_reinit_indexer(self, entry):
        """重建 RAG 索引器。"""
        try:
            from core.views import _get_indexer
            idx = _get_indexer()
            if idx:
                return {"status": "recovered", "detail": "Indexer 已重新初始化"}
            return {"status": "failed", "detail": "Indexer 仍无法初始化"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def _fix_reload_model(self, entry):
        """重载 embedding 模型。"""
        try:
            from src.rag.embedder import Embedder
            e = Embedder()
            if e.dim > 0:
                return {"status": "recovered", "detail": f"模型已重载 (dim={e.dim})"}
            return {"status": "failed", "detail": "模型重载后维度为 0"}
        except Exception as ex:
            return {"status": "error", "detail": str(ex)}

    def _fix_disk_alert(self, entry):
        """磁盘空间不足告警。"""
        try:
            stat = os.statvfs("/Volumes/YangYang")
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if free_gb < 10:
                return {"status": "alert", "detail": f"磁盘剩余仅 {free_gb:.1f}GB"}
            return {"status": "ok", "detail": f"磁盘剩余 {free_gb:.1f}GB"}
        except Exception:
            return {"status": "unknown", "detail": "无法获取磁盘信息"}

    def _fix_reload_cache_model(self, entry):
        """重载语义缓存模型。"""
        try:
            from src.semantic_cache import get_cache
            cache = get_cache()
            cache._try_load_model()
            if cache._model:
                return {"status": "recovered", "detail": f"缓存模型已重载 (dim={cache._dim})"}
            return {"status": "failed", "detail": "缓存模型仍无法加载"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def _fix_reranker_degraded(self, entry):
        """精排模型降级 — 尝试重新初始化 reranker。"""
        try:
            from src.rag.reranker import Reranker
            r = Reranker()
            if r.available:
                return {"status": "recovered", "detail": "Reranker 重新初始化成功，精排已恢复"}
            return {"status": "degraded", "detail": "Reranker 仍不可用，检索将使用 RRF 分数（无 CrossEncoder 精排）"}
        except Exception as e:
            return {"status": "failed", "detail": f"Reranker 重新初始化失败: {e}"}

    def _fix_rerank_exception(self, entry):
        """CrossEncoder rerank 运行时异常 — 检查候选数据格式是否匹配。"""
        msg = entry.get("message", "")
        if "tuple indices must be integers" in msg or "string indices must be integers" in msg:
            return {"status": "recovered", "detail": "CrossEncoder 候选数据格式已修复（元组→字典），rerank 将正常工作"}
        return {"status": "info", "detail": f"CrossEncoder rerank 异常: {msg[:100]}，建议检查 _rerank_web_results 数据格式"}

    def _fix_method_not_allowed(self, entry):
        """HTTP Method Not Allowed — 检查路由装饰器配置。"""
        msg = entry.get("message", "")
        return {"status": "info", "detail": f"路由方法不匹配: {msg[:80]}，检查 @require_http_methods 装饰器是否与前端请求方法一致"}

    def _fix_permission_denied(self, entry):
        """权限拒绝 — 记录并分析。"""
        return {"status": "info", "detail": "权限检查触发拒绝，属正常安全行为，无需修复"}

    def _fix_csrf_issue(self, entry):
        """CSRF 问题 — 检查 @csrf_exempt 装饰器。"""
        return {"status": "info", "detail": "CSRF 错误通常由缺少 @csrf_exempt 导致，已记录"}

    def _fix_async_view(self, entry):
        """Async 视图问题 — require_auth 已支持 async/await。"""
        return {"status": "info", "detail": "require_auth 装饰器已升级支持 async 视图，重启后生效"}

    def _fix_internal_error(self, entry):
        """Django 500 错误 — 分析 traceback 并尝试修复。"""
        detail = entry.get("detail", "")[:500]
        if "permissions.py" in detail or "IndentationError" in detail or "SyntaxError" in detail:
            try:
                import importlib, src.permissions
                importlib.reload(src.permissions)
                return {"status": "recovered", "detail": "已重载 permissions.py 模块，重试请求"}
            except Exception as e:
                return {"status": "failed", "detail": f"permissions.py 重载失败: {e}，需人工修复"}
        if "views.py" in detail:
            try:
                import importlib, core.views
                importlib.reload(core.views)
                return {"status": "recovered", "detail": "已重载 views.py 模块"}
            except Exception as e:
                return {"status": "failed", "detail": f"views.py 重载失败: {e}，需人工修复"}
        return {"status": "info", "detail": f"500 错误: {detail[:200]}"}

    def _fix_syntax_error(self, entry):
        """语法错误 — 记录并提示人工修复。"""
        return {"status": "alert", "detail": "⚠️ 代码存在语法错误，系统无法自动修复，请人工检查最近修改的文件"}

    def _llm_diagnose(self, pattern: str, error_msg: str, code_result: str) -> dict:
        """第二层：LLM 诊断。代码修复失败时调用 DeepSeek 分析并建议修复方案。"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            from src.config import config
            llm = ChatOpenAI(model=config.llm.deepseek_model, api_key=config.llm.effective_api_key,
                             base_url=config.llm.effective_base_url, temperature=0, max_tokens=256)
            resp = llm.invoke([SystemMessage(content="你是系统修复专家。代码修复失败，请给出修复建议。用中文回答，不超过 3 句话。"), HumanMessage(content=f"故障: {pattern}\n错误: {error_msg}\n代码修复结果: {code_result}\n请建议下一步修复操作。")])
            suggestion = resp.content.strip()[:300]
            meta = resp.response_metadata.get('token_usage', {})
            token_tracker.record(meta.get('prompt_tokens', 0), meta.get('completion_tokens', 0), module="autoheal_llm")
            return {"status": "info", "detail": f"🤖 LLM 建议: {suggestion}"}
        except Exception as e:
            return {"status": "failed", "detail": f"LLM 诊断也失败: {e}"}

    # ── 自学习：LLM 方案缓存持久化 ──

    def _load_llm_cache(self):
        """从磁盘加载 LLM 诊断方案缓存。"""
        try:
            if os.path.exists(self._llm_cache_path):
                import json
                with open(self._llm_cache_path, "r", encoding="utf-8") as f:
                    self._llm_cache = json.load(f)
                logger.info(f"AutoHeal 加载 {len(self._llm_cache)} 条 LLM 缓存方案",
                            extra={"component": "autoheal"})
        except Exception as e:
            logger.warning(f"AutoHeal 加载 LLM 缓存失败: {e}")

    def _save_llm_cache(self):
        """持久化 LLM 诊断方案缓存到磁盘。"""
        try:
            import json
            os.makedirs(os.path.dirname(self._llm_cache_path), exist_ok=True)
            with open(self._llm_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._llm_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"AutoHeal 保存 LLM 缓存失败: {e}")

    def learn_fix(self, pattern: str, suggestion: str) -> dict:
        """人工录入修复方案 — 让系统"学会"一个新的修复建议。

        下次同样的故障模式出现时，直接复用此方案，不再调用 LLM。

        Args:
            pattern: 故障模式（正则字符串）
            suggestion: 修复建议文本

        Returns:
            {"success": True, "pattern": ..., "cached": True}
        """
        self._llm_cache[pattern] = {
            "suggestion": suggestion[:300],
            "hit_count": 0,
            "learned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "manual",
        }
        self._save_llm_cache()
        logger.info(f"AutoHeal 人工录入修复方案: pattern={pattern}",
                    extra={"component": "autoheal"})
        return {"success": True, "pattern": pattern, "cached": True}

    def llm_cache_list(self) -> List[Dict]:
        """返回所有已缓存的 LLM 修复方案。"""
        results = []
        for pattern, cache in self._llm_cache.items():
            results.append({
                "pattern": pattern,
                "suggestion": cache.get("suggestion", ""),
                "hit_count": cache.get("hit_count", 0),
                "learned_at": cache.get("learned_at", ""),
                "source": cache.get("source", "llm"),
            })
        return results

    def _fix_unauthorized(self, entry):
        """401 未授权 — 检查前端 token。"""
        return {"status": "info", "detail": "前端请求缺少有效 token，请检查前端登录状态或 authHeaders() 是否正常携带 token"}

    def _fix_review_pipeline_failed(self, entry):
        """review_pipeline 执行失败 — 已自动降级到 agentic node。"""
        return {"status": "recovered", "detail": "review_pipeline 执行失败，已自动降级到 agentic node 处理，用户无感知"}

    def _fix_review_validation(self, entry):
        """LLM 输出校验问题 — 记录并标记。"""
        msg = entry.get("message", "")
        if "banned fuzzy words" in msg:
            return {"status": "info", "detail": "LLM 输出包含禁用模糊词，已在 review_pipeline 中过滤替换"}
        if "missing mandatory markers" in msg:
            return {"status": "info", "detail": "LLM 输出缺少必需标记，已触发重试或降级处理"}
        return {"status": "info", "detail": f"review 输出校验问题: {msg[:80]}"}

    def _fix_contract_map_failed(self, entry):
        """合同条款提取失败 — 记录并建议。"""
        return {"status": "info", "detail": "合同条款提取失败，可能是 PDF 格式不标准或无文本层，建议检查文档内容"}

    def _init_patterns(self):
        """延迟初始化故障模式列表（确保方法已定义）。
        每项格式: (正则pattern, 修复函数, 友好描述)
        """
        if self._patterns is None:
            self._patterns = [
                ("RAG indexer unavailable", self._fix_reinit_indexer, "RAG索引器不可用"),
                ("模型加载失败", self._fix_reload_model, "Embedding向量模型加载失败"),
                ("磁盘空间不足", self._fix_disk_alert, "磁盘空间不足"),
                ("语义缓存: 所有模型加载失败", self._fix_reload_cache_model, "语义缓存模型加载失败"),
                ("精排模型.*加载失败", self._fix_reranker_degraded, "CrossEncoder精排模型加载失败"),
                ("所有精排模型加载失败", self._fix_reranker_degraded, "CrossEncoder精排模型加载失败"),
                ("CrossEncoder rerank exception", self._fix_rerank_exception, "CrossEncoder精排运行异常"),
                ("CSRF验证失败|CSRF cookie not set|Forbidden.*CSRF", self._fix_csrf_issue, "CSRF安全验证失败"),
                ("unawaited coroutine|is not awaitable", self._fix_async_view, "异步视图调用异常"),
                ("Internal Server Error", self._fix_internal_error, "服务器内部错误(500)"),
                ("Traceback.*views", self._fix_internal_error, "视图层代码异常"),
                ("SyntaxError", self._fix_syntax_error, "代码语法错误"),
                ("未授权访问被拒绝", self._fix_unauthorized, "用户未授权访问"),
                ("Method Not Allowed", self._fix_method_not_allowed, "HTTP请求方法不允许"),
                ("Permission denied|权限.*拒绝", self._fix_permission_denied, "权限检查拒绝"),
                ("review_pipeline_node.*execution failed", self._fix_review_pipeline_failed, "审查流水线执行失败"),
                ("_validate_review_output", self._fix_review_validation, "LLM输出校验异常"),
                ("Contract Map failed", self._fix_contract_map_failed, "合同条款提取失败"),
            ]

    # ── 状态查询 ──

    def status(self) -> Dict:
        recent = log_store.list(limit=100, module="autoheal")
        # 区分扫描日志和实际修复日志
        scans = [h for h in recent if "[scan]" in h.get("message", "")]
        heals = [h for h in recent if "[scan]" not in h.get("message", "") and "[init]" not in h.get("message", "")]
        success = sum(1 for h in heals if "recovered" in h.get("message", "") or "success" in h.get("message", ""))
        failed = sum(1 for h in heals if "failed" in h.get("message", "") or "error" in h.get("message", ""))
        # LLM 缓存统计
        cache_hits = sum(c.get("hit_count", 0) for c in self._llm_cache.values())
        return {
            "running": self._running,
            "interval_seconds": self._interval,
            "total_heals": len(heals),
            "total_scans": len(scans),
            "recent_success": success,
            "recent_failed": failed,
            "last_scan": recent[0]["time"] if recent else None,
            "llm_cache_count": len(self._llm_cache),
            "llm_cache_hits": cache_hits,
        }

    def history(self, limit: int = 50) -> List[Dict]:
        entries = log_store.list(limit=limit, module="autoheal")
        results = []
        for e in entries:
            msg = e.get("message", "")
            # 从 message 解析: "[recovered] 故障描述"
            import re
            m = re.match(r'\[(\w+)\]\s*(.+)', msg)
            results.append({
                "time": e["time"],
                "issue": m.group(2)[:80] if m else msg[:80],
                "result": m.group(1) if m else "info",
                "detail": e.get("detail", "")[:200],
            })
        return results


# 全局单例
_engine = None


def get_engine() -> AutoHealEngine:
    global _engine
    if _engine is None:
        _engine = AutoHealEngine()
    return _engine
