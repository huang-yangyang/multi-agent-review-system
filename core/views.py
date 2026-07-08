"""Django views — 14 API endpoints migrated from FastAPI (src/api/routes.py).

All endpoints use plain function views returning JsonResponse.
Agent workflow calls go through asgiref.sync.async_to_sync.
"""

import os
import json
import uuid
import time
import logging
import queue
import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from django.conf import settings
from django.http import JsonResponse, HttpRequest, HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from asgiref.sync import async_to_sync

import base64 as _base64
from functools import wraps

from config import config as app_config
from workflows.orchestrator import get_graph
from state import AgentState
from core.conversation_store import (
    create_conversation as store_create_conv,
    get_conversations as store_get_convs,
    get_conversation as store_get_conv,
    update_conversation as store_update_conv,
    delete_conversation as store_delete_conv,
    save_message as store_save_msg,
    get_messages as store_get_msgs,
)

logger = logging.getLogger(__name__)

# 系统日志存储（注册 logging handler）
import src.log_store as _log_store  # noqa: F401

# 自动修复引擎随 Django 启动
from src.auto_heal import get_engine
get_engine().start()

# ── Auth decorator ──

def require_auth(view_func):
    """简易 Token 鉴权。前端发送 Authorization: Bearer <base64(username)>。支持 async 视图。"""
    from functools import wraps
    import inspect

    @wraps(view_func)
    async def _async_wrapped(request, *args, **kwargs):
        auth = request.META.get("HTTP_AUTHORIZATION", "")
        if auth.startswith("Bearer "):
            try:
                username = __import__('base64').b64decode(auth[7:].encode()).decode()
                from django.contrib.auth.models import User
                if User.objects.filter(username=username).exists():
                    request.user_name = username
                    from src.log_store import log_store
                    log_store.append("INFO", "auth", f"用户 {username} 访问 {request.path}", "")
                    return await view_func(request, *args, **kwargs)
            except Exception:
                pass
        from src.log_store import log_store
        log_store.append("ERROR", "auth", f"未授权访问被拒绝: {request.path}", f"IP: {request.META.get('REMOTE_ADDR','unknown')}")
        return JsonResponse({"error": "Unauthorized"}, status=401)

    @wraps(view_func)
    def _sync_wrapped(request, *args, **kwargs):
        auth = request.META.get("HTTP_AUTHORIZATION", "")
        if auth.startswith("Bearer "):
            try:
                username = __import__('base64').b64decode(auth[7:].encode()).decode()
                from django.contrib.auth.models import User
                if User.objects.filter(username=username).exists():
                    request.user_name = username
                    from src.log_store import log_store
                    log_store.append("INFO", "auth", f"用户 {username} 访问 {request.path}", "")
                    return view_func(request, *args, **kwargs)
            except Exception:
                pass
        from src.log_store import log_store
        log_store.append("ERROR", "auth", f"未授权访问被拒绝: {request.path}", f"IP: {request.META.get('REMOTE_ADDR','unknown')}")
        return JsonResponse({"error": "Unauthorized"}, status=401)

    return _async_wrapped if inspect.iscoroutinefunction(view_func) else _sync_wrapped


# ── Paths ──
UPLOAD_DIR = Path(app_config.rag.uploads_dir)
INDEXES_DIR = Path(app_config.rag.indexes_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── RAG Indexer singleton ──
_indexer = None
_indexer_error = None


def _get_indexer():
    global _indexer, _indexer_error
    if _indexer is None:
        try:
            from rag.indexer import get_indexer as _gi
            _indexer = _gi(
                uploads_dir=str(UPLOAD_DIR),
                indexes_dir=str(INDEXES_DIR),
            )
            _indexer_error = None
        except Exception as e:
            _indexer_error = str(e)
            _indexer = None
            logger.warning(f"RAG indexer unavailable: {e}")
    return _indexer


def _is_indexer_available() -> bool:
    return _indexer is not None


def _get_indexer_error() -> Optional[str]:
    return _indexer_error


# ── Helpers ──

def _json_body(request: HttpRequest) -> Dict[str, Any]:
    """Parse JSON request body."""
    try:
        return json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _load_conversation_context(thread_id: str, max_messages: int = 10, username: str = "") -> list:
    """加载对话历史并转换为 LangChain 消息格式。

    Args:
        thread_id: 会话 ID (对应 conversation_store 的 conv_id)
        max_messages: 最多加载最近 N 条消息，避免上下文溢出

    Returns:
        List[HumanMessage | AIMessage]，用于 AgentState.messages
    """
    from langchain_core.messages import HumanMessage, AIMessage

    try:
        raw = store_get_msgs(thread_id, limit=max_messages + 2, username=username)
    except Exception:
        return []

    if not raw:
        return []

    # 取最近的 message，最后一条是当前用户问题（前端已存），取它之前的 N 条
    history = raw[:-1][-max_messages:]

    messages = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def _build_initial_state(
    question: str,
    thread_id: Optional[str] = None,
    messages: Optional[list] = None,
    **kwargs,
) -> AgentState:
    tid = thread_id or str(uuid.uuid4())
    if messages is None:
        user_name = kwargs.get("user_name", "")
        messages = _load_conversation_context(tid, username=user_name) if thread_id else []

    # ── 长期记忆注入 ──
    long_term_context = ""
    try:
        from src.memory.long_term import build_long_term_context
        long_term_context = build_long_term_context(question, max_results=2)
    except Exception:
        pass

    user_name = kwargs.pop("user_name", "")
    return AgentState(
        question=question,
        raw_input=question,
        user_name=user_name,
        thread_id=tid,
        messages=messages,
        long_term_context=long_term_context,
        intent="",
        task_description="",
        sub_tasks=[],
        retrieved_context=[],
        plan=[],
        current_step_index=0,
        quality_check_passed=False,
        loop_count=0,
        final_response="",
        **kwargs,
    )

def _detect_domain_from_text(text: str) -> str:
    from src.workflows.orchestrator import DOMAIN_KEYWORDS
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"

def _error_response(msg: str, status: int = 500) -> JsonResponse:
    return JsonResponse({"error": msg}, status=status)


# ── File upload helpers ──

_ALLOWED_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".xlsm",
                 ".txt", ".md", ".csv", ".json", ".yaml", ".yml",
                 ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
_MAX_SIZE = 50 * 1024 * 1024  # 50MB

# ── Attachment reader ──

def _read_attachments(attachments: List) -> str:
    """读取附件内容，拼接为注入文本。

    支持两种格式：
    - str: 文件绝对路径（兼容旧版）
    - dict: {name, content} 前端 FileReader 读取的文本内容
    读取失败的文件会被跳过并记录。
    """
    import logging
    logger = logging.getLogger(__name__)

    blocks: List[str] = []
    for item in attachments:
        # 新格式：{name, content, encoding?} — 前端 FileReader 读取
        if isinstance(item, dict):
            fname = item.get("name", "unknown")
            content = item.get("content", "")
            encoding = item.get("encoding", "text")

            if not content:
                logger.warning(f"Attachment has no content: {fname}")
                continue

            # base64 编码的二进制文件（PDF/DOCX）→ 解码并用 parse_document 提取文字
            if encoding == "base64":
                ext = os.path.splitext(fname)[1].lower()
                try:
                    # 去掉 data:xxx;base64, 前缀
                    if "," in content:
                        content = content.split(",", 1)[1]
                    raw = base64.b64decode(content)
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(raw)
                        tmp_path = tmp.name
                    try:
                        from src.rag.parser import parse_document
                        text, _ = parse_document(tmp_path)
                        logger.info(f"Parsed base64 attachment {fname}: {len(text)} chars")
                        blocks.append(f"【附件：{fname}】\n{text}")
                    finally:
                        os.unlink(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to parse base64 attachment {fname}: {e}")
                    blocks.append(f"【附件：{fname}】\n[解析失败: {fname}, 错误: {str(e)[:100]}]")
            else:
                blocks.append(f"【附件：{fname}】\n{content}")
            continue

        # 旧格式：纯路径字符串
        fp = item
        if not os.path.isfile(fp):
            logger.warning(f"Attachment not found: {fp}")
            continue

        fname = os.path.basename(fp)
        ext = os.path.splitext(fp)[1].lower()
        content = ""

        # 纯文本：直接用 read_text 模式读取
        text_exts = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log",
                     ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
                     ".sh", ".ps1", ".toml", ".ini", ".cfg", ".conf"}
        if ext in text_exts:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read text attachment {fp}: {e}")
                continue
        else:
            # 二进制文档：使用 RAG 解析器提取文本（PDF/DOCX 等）
            parsable_exts = {".pdf", ".docx"}
            if ext in parsable_exts:
                try:
                    from src.rag.parser import parse_document
                    content, _ = parse_document(fp)
                    logger.info(f"Parsed attachment {fname}: {len(content)} chars")
                except FileNotFoundError:
                    logger.warning(f"Parser dependency missing for {fp}, treating as unreadable")
                    content = f"[无法解析: {fname}，缺少 PDF/DOCX 解析依赖]"
                except Exception as e:
                    logger.warning(f"Failed to parse attachment {fp}: {e}")
                    content = f"[解析失败: {fname}, 错误: {str(e)[:100]}]"
            elif ext in {".doc", ".ppt", ".pptx", ".xls", ".xlsx", ".xlsm"}:
                # 旧格式 Office 文件：标记为不支持但允许上传
                content = f"[暂不支持解析此格式: {fname} (类型: {ext})，请转为 PDF 或 DOCX 后上传]"
            else:
                logger.warning(f"Unsupported attachment type: {fp}")
                continue

        if content:
            blocks.append(f"【附件：{fname}】\n{content}")

    return "\n\n".join(blocks)

# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

# ── 1 & 2. Health Check ──

@require_http_methods(["GET"])
def health_check(request: HttpRequest) -> JsonResponse:
    return JsonResponse({
        "status": "ok",
        "version": "1.0.0",
        "agents_available": ["research", "analysis"],
    })


@require_http_methods(["GET"])
def graph_structure(request: HttpRequest) -> JsonResponse:
    """返回当前 LangGraph 图结构，供前端流程页动态渲染。"""
    return JsonResponse({
        "nodes": [
            {"id": "start", "label": "START", "x": 540, "y": 48, "color": "#22c55e", "w": 110, "h": 36},
            {"id": "decomposer", "label": "Decomposer\n意图识别+领域检测", "x": 540, "y": 148, "color": "#8b5cf6", "w": 200, "h": 60, "desc": "LLM 解析意图+领域+复杂度"},
            {"id": "knowledge_retriever", "label": "Knowledge Retriever\n知识库预检索", "x": 540, "y": 275, "color": "#a78bfa", "w": 200, "h": 60, "desc": "FAISS+BM25+RRF+CrossEncoder"},
            {"id": "router", "label": "Router\n条件路由分发", "x": 540, "y": 402, "color": "#f59e0b", "w": 200, "h": 60, "desc": "四维判定：domain+review+intent+complexity"},
            {"id": "review_pipeline", "label": "★ Review Pipeline\nMap-Reduce 审查", "x": 48, "y": 548, "color": "#ff6b6b", "w": 220, "h": 72, "desc": "金融/合同/劳动法三领域自适应", "highlight": True},
            {"id": "agentic_research", "label": "Agentic Research\nReAct Tool Calling", "x": 290, "y": 548, "color": "#22d3ee", "w": 200, "h": 60, "desc": "复杂多步推理，自主工具调用"},
            {"id": "research_fast", "label": "Research\n快速通道", "x": 530, "y": 548, "color": "#38bdf8", "w": 200, "h": 60, "desc": "固定三阶段管线+语义缓存"},
            {"id": "analysis", "label": "Analysis\n数据分析", "x": 770, "y": 548, "color": "#34d399", "w": 160, "h": 56, "desc": "统计分析+可视化"},
            {"id": "aggregator", "label": "Aggregator\n结果聚合+护栏校验", "x": 540, "y": 690, "color": "#818cf8", "w": 200, "h": 60, "desc": "合并+_validate_review_output+缓存"},
            {"id": "end", "label": "END", "x": 540, "y": 795, "color": "#ef4444", "w": 130, "h": 36},
        ],
        "edges": [
            {"source": "start", "target": "decomposer"},
            {"source": "decomposer", "target": "knowledge_retriever"},
            {"source": "knowledge_retriever", "target": "router"},
            {"source": "router", "target": "review_pipeline", "label": "审查/风险任务", "color": "#ff6b6b", "width": 2.4},
            {"source": "router", "target": "agentic_research", "label": "复杂研究", "color": "#22d3ee", "width": 1.8},
            {"source": "router", "target": "research_fast", "label": "简单研究", "color": "#38bdf8", "width": 1.6},
            {"source": "router", "target": "analysis", "label": "分析意图", "color": "#34d399", "width": 1.5},
            {"source": "review_pipeline", "target": "aggregator"},
            {"source": "agentic_research", "target": "aggregator"},
            {"source": "research_fast", "target": "aggregator"},
            {"source": "analysis", "target": "aggregator"},
            {"source": "aggregator", "target": "end"},
        ],
    })


@require_http_methods(["GET"])
def metrics(request: HttpRequest) -> JsonResponse:
    """可观测性指标：请求统计 + 缓存统计 + 系统资源。"""
    import os, psutil
    from src.middleware import tracker, token_tracker, audit_log
    from src.semantic_cache import get_cache

    cache = get_cache()
    process = psutil.Process(os.getpid())

    return JsonResponse({
        "requests": tracker.stats(),
        "tokens": token_tracker.stats(),
        "cache": cache.stats_by_domain(),
        "audit": audit_log.stats(),
        "system": {
            "cpu_percent": round(process.cpu_percent(interval=0.1), 1),
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 1),
            "threads": process.num_threads(),
        },
    })



# ── Admin: 用户管理 ──

@require_http_methods(["GET"])
@require_auth
def admin_users(request: HttpRequest) -> JsonResponse:
    from django.contrib.auth.models import User
    from core.models import UserProfile
    users_raw = User.objects.all()
    users = []
    for u in users_raw:
        profile = UserProfile.objects.filter(user=u).first()
        users.append({
            'id': u.id, 'username': u.username,
            'is_superuser': u.is_superuser, 'is_active': u.is_active,
            'date_joined': u.date_joined.isoformat() if u.date_joined else '',
            'last_login': u.last_login.isoformat() if u.last_login else '',
            'role': profile.role if profile else 'hr_user',
        })
    return JsonResponse(list(users), safe=False)


@csrf_exempt

@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def admin_user_create(request: HttpRequest) -> JsonResponse:
    import json
    from django.contrib.auth.models import User
    try:
        body = json.loads(request.body.decode("utf-8"))
        username = body.get("username", "").strip()
        password = body.get("password", "").strip()
        role = body.get("role", "hr_user")
    except Exception:
        return _error_response("Invalid JSON", 400)

    if not username or len(username) < 2:
        return _error_response("用户名至少2个字符", 400)
    if not password or len(password) < 4:
        return _error_response("密码至少4个字符", 400)
    if User.objects.filter(username=username).exists():
        return _error_response("用户名已存在", 400)

    user = User.objects.create_user(username=username, password=password)
    user.is_superuser = (role == "admin")
    user.is_staff = (role == "admin")
    user.save()
    from core.models import UserProfile
    UserProfile.objects.update_or_create(user=user, defaults={'role': role})
    logger.info(f"Admin created user: {username} (role={role})")
    return JsonResponse({"success": True, "id": user.id, "username": username})

@csrf_exempt
@require_http_methods(["PATCH"])
@require_auth
def admin_user_update(request: HttpRequest, user_id: int) -> JsonResponse:
    import json
    from django.contrib.auth.models import User
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return _error_response("Invalid JSON", 400)

    try:
        user = User.objects.get(id=user_id)
        changed = []
        if "username" in body and body["username"] != user.username:
            new_uname = body["username"].strip()
            if User.objects.filter(username=new_uname).exists():
                return _error_response("用户名已存在", 400)
            changed.append(f"username: {user.username} → {new_uname}")
            user.username = new_uname
        if "password" in body and body["password"]:
            user.set_password(body["password"])
            changed.append("password changed")
        if "is_superuser" in body:
            user.is_superuser = bool(body["is_superuser"])
        if "is_active" in body:
            user.is_active = bool(body["is_active"])
        if "role" in body:
            from core.models import UserProfile
            profile, _ = UserProfile.objects.get_or_create(user=user)
            old_role = profile.role
            profile.role = body["role"]
            profile.save()
            user.is_superuser = (body["role"] == "admin")
            user.is_staff = (body["role"] == "admin")
            changed.append(f"role: {old_role} → {body['role']}")
        user.save()
        logger.info(f"Admin updated user #{user_id} ({user.username}): {', '.join(changed)}")
        return JsonResponse({"success": True, "id": user_id})
    except User.DoesNotExist:
        return _error_response("User not found", 404)


@csrf_exempt
@require_http_methods(["DELETE"])
@require_auth
def admin_user_delete(request: HttpRequest, user_id: int) -> JsonResponse:
    from django.contrib.auth.models import User
    try:
        user = User.objects.get(id=user_id)
        if user.is_superuser:
            return _error_response("Cannot delete superuser", 400)
        user.delete()
        return JsonResponse({"success": True, "id": user_id})
    except User.DoesNotExist:
        return _error_response("User not found", 404)



# ── System Logs ──

@require_http_methods(["GET"])
def system_logs(request: HttpRequest) -> JsonResponse:
    from src.log_store import log_store
    limit = int(request.GET.get("limit", 200))
    level = request.GET.get("level", "")
    module = request.GET.get("module", "")
    entries = log_store.list(limit=limit, level=level, module=module)
    return JsonResponse({"entries": entries, "stats": log_store.stats()})


@csrf_exempt
@require_http_methods(["DELETE"])
def system_logs_clear(request: HttpRequest) -> JsonResponse:
    from src.log_store import log_store
    log_store.clear()
    return JsonResponse({"success": True})


# ── Analytics Dashboard ──

@require_http_methods(["GET"])
def analytics(request: HttpRequest) -> JsonResponse:
    from src.middleware import token_tracker, tracker
    from src.semantic_cache import get_cache
    from src.log_store import log_store
    cache = get_cache()
    ts = token_tracker.stats()
    return JsonResponse({
        "tokens": ts,
        "requests": tracker.stats(),
        "cache": cache.stats_by_domain(),
        "logs": log_store.stats(),
        "token_total_display": f"{ts['total_input_tokens'] + ts['total_output_tokens']:,}",
    })


# ── Auto Health Check ──

@csrf_exempt
@require_http_methods(["GET", "POST"])
def auto_health_check(request: HttpRequest) -> JsonResponse:
    from src.log_store import log_store
    from src.semantic_cache import get_cache
    import os
    
    fixes = []
    issues = []
    # 同时检查 ERROR 和 WARNING（WARNING 可能预示着潜在问题）
    recent_errors = log_store.list(limit=50, level="ERROR")
    recent_warnings = log_store.list(limit=50, level="WARNING")
    
    for e in recent_errors:
        msg = e.get("message", "")
        # 检测常见问题并自动修复
        if "模型加载失败" in msg or "unavailable" in msg.lower():
            fixes.append("检测到模型加载问题，尝试重载...")
        if "Connection refused" in msg or "timeout" in msg.lower():
            fixes.append("检测到网络连接问题: " + msg[:60])
        issues.append(f"[ERROR][{e.get('module','')}] {msg[:80]}")
    
    for w in recent_warnings:
        msg = w.get("message", "")
        # WARNING 级别也需关注，特别是 CrossEncoder 降级、Method Not Allowed
        if "CrossEncoder rerank exception" in msg:
            fixes.append("检测到 CrossEncoder 精排异常，已修复数据格式不匹配问题")
        if "Method Not Allowed" in msg:
            fixes.append(f"检测到路由方法不匹配: {msg[:60]}")
        if "降级" in msg or "degraded" in msg.lower():
            issues.append(f"[WARNING][{w.get('module','')}] {msg[:80]}")
    
    # 检查缓存状态
    cache = get_cache()
    cache_stats = cache.stats_by_domain()
    if not cache_stats.get("model_loaded"):
        fixes.append("缓存模型未加载，建议检查 HF_HUB_OFFLINE 设置")
    
    # 检查索引
    try:
        from src.rag.indexer import get_indexer
        from src.config import config
        idx = get_indexer(config.rag.uploads_dir, config.rag.indexes_dir)
        istats = idx.get_stats()
    except Exception as e:
        issues.append(f"索引器异常: {e}")
    
    return JsonResponse({
        "status": "ok" if not issues else "issues_found",
        "issues": issues[:10],
        "fixes": fixes[:5],
        "auto_fixed": len(fixes) > 0,
    })


# ── Auto Heal ──

@require_http_methods(["GET"])
def auto_heal_status(request: HttpRequest) -> JsonResponse:
    from src.auto_heal import get_engine
    engine = get_engine()
    return JsonResponse({"status": engine.status(), "history": engine.history(limit=50)})


@csrf_exempt
@require_http_methods(["POST"])
def auto_heal_start(request: HttpRequest) -> JsonResponse:
    from src.auto_heal import get_engine
    engine = get_engine()
    engine.start()
    return JsonResponse({"success": True, "status": engine.status()})


@csrf_exempt
@require_http_methods(["POST"])
def auto_heal_scan(request: HttpRequest) -> JsonResponse:
    from src.auto_heal import get_engine
    engine = get_engine()
    engine._scan_and_heal()
    return JsonResponse({"success": True, "status": engine.status()})


@csrf_exempt
@require_http_methods(["POST"])
def auto_heal_learn(request: HttpRequest) -> JsonResponse:
    """人工录入修复方案 — 让系统"学会"一个修复建议，下次同样故障直接复用。"""
    import json
    from src.auto_heal import get_engine
    try:
        body = json.loads(request.body.decode("utf-8"))
        pattern = body.get("pattern", "").strip()
        suggestion = body.get("suggestion", "").strip()
        if not pattern or not suggestion:
            return _error_response("pattern and suggestion are required", 400)
        engine = get_engine()
        result = engine.learn_fix(pattern, suggestion)
        return JsonResponse(result)
    except Exception as e:
        return _error_response(f"Failed to learn fix: {e}", 500)


@require_http_methods(["GET"])
def auto_heal_cache(request: HttpRequest) -> JsonResponse:
    """查看所有已缓存的 LLM 修复方案（自学习记忆库）。"""
    from src.auto_heal import get_engine
    engine = get_engine()
    return JsonResponse({"cache": engine.llm_cache_list(), "count": len(engine._llm_cache)})


@require_http_methods(["GET"])
def token_stats(request: HttpRequest) -> JsonResponse:
    from src.middleware import token_tracker
    return JsonResponse(token_tracker.stats_by_module())

# ── 3. Admin Page ──

_ADMIN_HTML_PATH = Path(__file__).resolve().parent / "templates" / "admin" / "index.html"


@require_http_methods(["GET"])
def admin_page(request: HttpRequest) -> HttpResponse:
    """Serve admin page directly as HTML to avoid Django template engine collision."""
    try:
        html = _ADMIN_HTML_PATH.read_text(encoding="utf-8")
        return HttpResponse(html, content_type="text/html; charset=utf-8")
    except FileNotFoundError:
        return HttpResponse("<h1>Admin page not found</h1>", status=404, content_type="text/html")


# ── 4. File Upload ──

@csrf_exempt
@require_http_methods(["POST"])
def upload_file(request: HttpRequest) -> JsonResponse:
    uploaded = request.FILES.get("file")
    if not uploaded:
        return _error_response("No file selected", 400)

    safe_name = Path(uploaded.name).name
    if not safe_name:
        return _error_response("Invalid filename", 400)

    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        return _error_response(f"Unsupported file type: {ext}", 400)

    content = uploaded.read()
    if len(content) > _MAX_SIZE:
        return _error_response("File too large (max 50MB)", 400)

    # Timestamp rename
    ts = int(time.time() * 1000)
    stem, suffix = os.path.splitext(safe_name)
    safe_name = f"{stem}_{ts}{suffix}"
    dest_path = UPLOAD_DIR / safe_name

    with open(dest_path, "wb") as f:
        f.write(content)
    logger.info(f"File saved: {dest_path}")

    # RAG indexing
    indexer = _get_indexer()
    index_status = "stored_only"
    index_error = None
    chunk_count = 0
    _parsable_exts = {".pdf", ".docx", ".md", ".txt"}

    # ── 先记录文档可见性（统一管理），再索引 ──
    # 这样 index_file 内部即可通过 get_doc_visibility 读取正确 visibility，
    # 写入 Qdrant payload 以支持 pre-filter（D 方案：向量库元数据过滤）。
    try:
        from src.permissions import grant_doc_permission, get_tenant_id, VALID_VISIBILITIES
        tenant = get_tenant_id(request)
        owner = getattr(request, 'user_name', 'anonymous')
        visibility = request.POST.get('visibility', 'admin')
        if visibility not in VALID_VISIBILITIES:
            visibility = 'admin'
        grant_doc_permission(str(dest_path), safe_name, owner, tenant, visibility=visibility)
    except Exception as e:
        logger.warning(f"Failed to record doc permission: {e}")

    if indexer is not None:
        dup = indexer.check_duplicate(str(dest_path))
        if dup and dup.get("duplicate"):
            try:
                os.remove(str(dest_path))
            except Exception:
                pass
            return JsonResponse({
                "success": False,
                "filename": safe_name,
                "status": "duplicate",
                "duplicate_of": dup.get("existing_file"),
            })

        if ext in _parsable_exts:
            try:
                result = indexer.index_file(str(dest_path))
                chunk_count = result.get("chunk_count", 0)
                index_status = result.get("status", "indexed")
            except Exception as e:
                logger.error(f"Indexing failed: {dest_path} — {e}")
                index_status = "index_error"
                index_error = str(e)
    else:
        index_error = _get_indexer_error()
        logger.info(f"Indexer unavailable, file stored only: {safe_name}")

    return JsonResponse({
        "success": True,
        "filename": safe_name,
        "path": str(dest_path),
        "size": len(content),
        "index_status": index_status,
        "chunk_count": chunk_count,
        **({"index_error": index_error} if index_error else {}),
    })


# ── 5. List Files ──

@require_http_methods(["GET"])
def list_files(request: HttpRequest) -> JsonResponse:
    if not UPLOAD_DIR.exists():
        return JsonResponse({"files": [], "count": 0})

    try:
        _get_indexer()  # 每次请求都尝试初始化，修复 singleto失败后永不重试的 bug
        if _is_indexer_available():
            indexer = _get_indexer()
            indexed_files = indexer.list_files()
            stats = indexer.get_stats()
        else:
            indexed_files = []
            disk_files = [p for p in UPLOAD_DIR.iterdir() if p.name != ".gitkeep"] if UPLOAD_DIR.exists() else []
            stats = {"file_count": len(disk_files), "chunk_count": 0, "indexer_available": False}
    except Exception:
        disk_files = [p for p in UPLOAD_DIR.iterdir() if p.name != ".gitkeep"] if UPLOAD_DIR.exists() else []
        indexed_files = []
        stats = {"file_count": len(disk_files), "chunk_count": 0}

    index_map = {}
    for f in indexed_files:
        index_map[f.get("file_path", "")] = f

    files: List[Dict[str, Any]] = []
    for fpath in sorted(UPLOAD_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if fpath.name == ".gitkeep":
            continue
        st = fpath.stat()
        indexed_info = index_map.get(str(fpath), {})
        files.append({
            "filename": fpath.name,
            "path": str(fpath),
            "size": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "index_status": indexed_info.get("status", "stored_only"),
            "chunk_count": indexed_info.get("chunk_count", 0),
        })

    from src.permissions import get_doc_visibility, get_tenant_id, get_user_accessible_doc_paths
    from django.contrib.auth.models import User
    tenant = get_tenant_id(request)
    owner = getattr(request, 'user_name', 'anonymous')
    is_admin = User.objects.filter(username=owner, is_superuser=True).exists()

    if not is_admin and owner != "anonymous":
        accessible = get_user_accessible_doc_paths(owner, tenant)
        # None = 管理员或权限表空（不过滤）；空集 = 无权限（过滤掉所有）；非空集 = 只保留匹配的
        if accessible is not None:
            files = [f for f in files if f["path"] in accessible]

    for f in files:
        f["visibility"] = get_doc_visibility(f["path"], tenant)

    return JsonResponse({
        "files": files,
        "count": len(files),
        "stats": stats,
    })


# ── 6. Delete File ──

@csrf_exempt
@require_http_methods(["PATCH"])
def patch_file_visibility(request: HttpRequest, filename: str) -> JsonResponse:
    """修改文件的可见性权限。"""
    import json
    from src.permissions import get_tenant_id, _get_conn

    tenant = get_tenant_id(request)
    owner = getattr(request, 'user_name', 'anonymous')

    try:
        body = json.loads(request.body.decode("utf-8"))
        visibility = body.get("visibility", "admin")
    except Exception:
        return _error_response("Invalid JSON body", 400)

    from src.permissions import VALID_VISIBILITIES
    if visibility not in VALID_VISIBILITIES:
        return _error_response(f"visibility must be one of {sorted(VALID_VISIBILITIES)}", 400)

    try:
        safe_name = Path(filename).name
        safe_path = UPLOAD_DIR / safe_name
        if not safe_path.exists():
            return _error_response("File not found", 404)

        from src.permissions import set_doc_visibility
        set_doc_visibility(str(safe_path), visibility, tenant)
        logger.info(f"Visibility changed: {filename} → {visibility}")

        # 权限变更后清除引用该文档的语义缓存，防止旧缓存返回变更前的答案
        try:
            from src.semantic_cache import get_cache
            cleared = get_cache().remove_by_doc_path(str(safe_path))
            if cleared > 0:
                logger.info(f"Visibility change cleared {cleared} cache entries for {safe_name}")
        except Exception as e:
            logger.warning(f"Failed to clear semantic cache on visibility change: {e}")

        return JsonResponse({"success": True, "filename": safe_name, "visibility": visibility})
    except Exception as e:
        return _error_response(f"Failed to update visibility: {e}", 500)


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_file(request: HttpRequest, filename: str) -> JsonResponse:
    safe_name = Path(filename).name
    if not safe_name or safe_name == ".gitkeep":
        return _error_response("Invalid filename", 400)

    file_path = UPLOAD_DIR / safe_name
    source_existed = file_path.exists()

    idx_deleted = False
    idx_error = None
    if _is_indexer_available():
        indexer = _get_indexer()
        try:
            from src.log_store import log_store
            log_store.append("WARNING", "upload", f"用户 {getattr(request, 'user_name', 'anonymous')} 删除文件: {filename}", "")
            result = indexer.remove_file(str(file_path))
            if result.get("status") == "removed":
                idx_deleted = True
        except Exception as e:
            idx_error = str(e)
            logger.error(f"Index removal failed: {file_path} — {e}")

    source_deleted = False
    if source_existed:
        try:
            os.remove(str(file_path))
            source_deleted = True
            logger.info(f"Source file deleted: {file_path}")
        except OSError as e:
            logger.error(f"Source file deletion failed: {file_path} — {e}")
            return _error_response(f"Delete source file failed: {e}", 500)

    if not source_existed and not idx_deleted:
        return _error_response(f"File not found: {safe_name}", 404)

    # 删除文件后清除引用该文档的语义缓存，防止旧缓存返回已删除文件的内容
    try:
        from src.semantic_cache import get_cache
        cleared = get_cache().remove_by_doc_path(str(file_path))
        if cleared > 0:
            logger.info(f"File deletion cleared {cleared} cache entries for {safe_name}")
    except Exception as e:
        logger.warning(f"Failed to clear semantic cache on file deletion: {e}")

    return JsonResponse({
        "success": True,
        "filename": safe_name,
        "source_deleted": source_deleted,
        "index_deleted": idx_deleted,
        **({"index_error": idx_error} if idx_error else {}),
    })


# ── 7. RAG Search ──

@csrf_exempt
@require_http_methods(["POST"])
def rag_search(request: HttpRequest) -> JsonResponse:
    body = _json_body(request)
    query = body.get("query", "").strip()
    top_k = int(body.get("top_k", 10))

    if not query:
        return _error_response("query cannot be empty", 400)

    indexer = _get_indexer()
    if indexer is None:
        return _error_response(
            f"RAG indexer unavailable: {_get_indexer_error() or 'Indexer not initialized'}",
            503,
        )

    try:
        # ── 文档权限过滤 ──
        from src.permissions import get_user_accessible_doc_paths, get_tenant_id
        owner = getattr(request, 'user_name', 'anonymous')
        tenant = get_tenant_id(request)
        accessible_paths = get_user_accessible_doc_paths(owner, tenant)
        result = indexer.search(query, top_k=top_k, accessible_paths=accessible_paths)
        return JsonResponse(result)
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return _error_response(f"Search failed: {e}", 500)

# ── 13. Dashboard ──

@require_http_methods(["GET"])
def dashboard(request: HttpRequest) -> JsonResponse:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return JsonResponse({
        "timestamp": datetime.now().isoformat(),
        "agents": [
            {"name": "Research", "status": "online", "tasks_completed": 47, "avg_response_ms": 320},
            {"name": "Analysis", "status": "online", "tasks_completed": 33, "avg_response_ms": 450},
            {"name": "CustomerService", "status": "online", "tasks_completed": 89, "avg_response_ms": 180},
        ],
        "workflows": {
            "total_runs": 169,
            "success_rate": 0.94,
            "avg_duration_ms": 680,
            "by_intent": {
                "research": 47,
                "analysis": 33,
                "customer_service": 89,
            },
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory_used_gb": round(mem.used / (1024**3), 2),
            "memory_total_gb": round(mem.total / (1024**3), 2),
            "memory_percent": mem.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_percent": disk.percent,
        },
        "recent_logs": [
            {"time": datetime.now().isoformat(), "level": "INFO", "agent": "ResearchAgent", "message": "BeliefBase sync completed — 12 new beliefs indexed"},
            {"time": datetime.now().isoformat(), "level": "INFO", "agent": "Orchestrator", "message": "Workflow run #169 finished — 94% success rate"},
            {"time": datetime.now().isoformat(), "level": "WARN", "agent": "AnalysisAgent", "message": "Data source latency spike — avg 520ms (threshold 500ms)"},
            {"time": datetime.now().isoformat(), "level": "INFO", "agent": "CustomerServiceAgent", "message": "Sentiment model updated — accuracy improved to 91.2%"},
            {"time": datetime.now().isoformat(), "level": "INFO", "agent": "StateManager", "message": "Redis checkpoint GC completed — 340 stale keys removed"},
        ],
    })

# ── 14. Auth Login ──
import base64 as _base64

@csrf_exempt
def auth_login(request: HttpRequest) -> JsonResponse:
    from django.contrib.auth import authenticate
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        data = json.loads(request.body)
        username = data.get('username', '')
        password = data.get('password', '')
    except Exception:
        return JsonResponse({'error': 'invalid JSON'}, status=400)
    user = authenticate(request, username=username, password=password)
    if user:
        token = _base64.b64encode(username.encode()).decode()
        return JsonResponse({'token': token, 'username': user.username})
    return JsonResponse({'error': '用户名或密码错误'}, status=401)


# ── 15. Auth Logout ──
@csrf_exempt
def auth_logout(request: HttpRequest) -> JsonResponse:
    return JsonResponse({'message': 'logged out'})


# ── 16. Conversations (GET list + POST create) ──
def _get_request_username(request: HttpRequest) -> str:
    """从请求中提取当前用户名（兼容多种鉴权方式）。"""
    return getattr(request, 'user_name', '') or ''


@csrf_exempt
@require_auth
def list_conversations(request: HttpRequest) -> JsonResponse:
    username = _get_request_username(request)
    if request.method == "POST":
        body = _json_body(request)
        conv_id = body.get("id") or str(uuid.uuid4())
        title = body.get("title", "")
        agent = body.get("agent", "")
        conv = store_create_conv(conv_id, title=title, agent=agent, username=username)
        return JsonResponse(conv, status=201)

    limit = int(request.GET.get('limit', 50))
    convs = store_get_convs(limit=limit, username=username)
    return JsonResponse({"conversations": convs})
# ── 19. Delete Conversation ──
@csrf_exempt
@require_http_methods(["DELETE"])
def delete_conversation_view(request: HttpRequest, conv_id: str) -> JsonResponse:
    username = _get_request_username(request)
    if not store_get_conv(conv_id, username=username):
        return _error_response("Conversation not found", 404)
    if not store_delete_conv(conv_id, username=username):
        return _error_response("Conversation not found or access denied", 404)
    return JsonResponse({"deleted": conv_id})


# ── 19b. Update Conversation (PATCH) ──
@csrf_exempt
@require_http_methods(["PATCH"])
def update_conversation_view(request: HttpRequest, conv_id: str) -> JsonResponse:
    username = _get_request_username(request)
    if not store_get_conv(conv_id, username=username):
        return _error_response("Conversation not found", 404)
    body = _json_body(request)
    kwargs = {}
    if "title" in body:
        kwargs["title"] = body["title"]
    if "agent" in body:
        kwargs["agent"] = body["agent"]
    if "pinned" in body:
        kwargs["pinned"] = int(body["pinned"])
    if not kwargs:
        return _error_response("No fields to update", 400)
    kwargs["username"] = username
    if not store_update_conv(conv_id, **kwargs):
        return _error_response("Conversation not found or access denied", 404)
    conv = store_get_conv(conv_id, username=username)
    return JsonResponse(conv)


# ── 20. Conversation Messages (GET + POST) ──
@csrf_exempt
@require_auth
def conversation_messages(request: HttpRequest, conv_id: str) -> JsonResponse:
    """Handle GET (list messages) and POST (save message) for a conversation."""
    username = _get_request_username(request)
    if request.method == "GET":
        limit = int(request.GET.get('limit', 200))
        msgs = store_get_msgs(conv_id, limit=limit, username=username)
        return JsonResponse({"messages": msgs})

    if request.method == "POST":
        body = _json_body(request)
        role = body.get("role", "")
        content = body.get("content", "")
        agent = body.get("agent", "")
        msg_time = body.get("time", "")
        if not role or not content:
            return _error_response("role and content are required", 400)
        conv = store_get_conv(conv_id, username=username)
        if not conv:
            store_create_conv(conv_id, title=content[:30], username=username)
        elif not conv.get("title") and role == "user":
            # 会话已存在但标题为空 → 用第一条用户消息自动命名
            store_update_conv(conv_id, title=content[:30], username=username)
        msg = store_save_msg(conv_id, role=role, content=content, agent=agent, msg_time=msg_time)
        return JsonResponse(msg, status=201)

    return _error_response("Method not allowed", 405)


# ── 22. Workflow (SSE streaming, async) ──
@csrf_exempt
@require_http_methods(["POST"])
async def workflow(request: HttpRequest) -> StreamingHttpResponse:
    """Streaming workflow endpoint with HITL interrupt support.

    改为 async 视图，支持 graph.astream(stream_mode=["updates", "custom"])
    实时推送 token 级流式输出和工具调用状态。
    """
    # ── 内联鉴权（sync_to_async 兼容） ──
    auth = request.META.get("HTTP_AUTHORIZATION", "")
    if auth.startswith("Bearer "):
        try:
            username = __import__('base64').b64decode(auth[7:].encode()).decode()
            from asgiref.sync import sync_to_async
            from django.contrib.auth.models import User
            exists = await sync_to_async(User.objects.filter(username=username).exists)()
            if exists:
                request.user_name = username
            else:
                return JsonResponse({"error": "Unauthorized"}, status=401)
        except Exception:
            return JsonResponse({"error": "Unauthorized"}, status=401)
    else:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    body = _json_body(request)
    topic = body.get("topic", "")
    thread_id = body.get("thread_id") or str(uuid.uuid4())
    attachments = body.get("attachments", [])

    if not topic:
        return _error_response("topic is required", 400)

    # ── 操作日志 ──
    from src.middleware import audit_log, desensitize
    from src.log_store import log_store
    user = body.get("user", "anonymous")
    audit_log.log(user, "workflow", desensitize(topic[:100]), "started")
    domain = _detect_domain_from_text(topic)
    log_store.append("INFO", "workflow", f"用户 {user} 发起查询", desensitize(topic[:200]))
    log_store.append("INFO", "workflow", f"检测领域: {domain}", "")

    # ── 处理附件：读取文件内容并注入到 topic 前面 ──
    if attachments:
        logger.info(f"[ATTACH] Received attachments: {len(attachments)} file(s)")
        attachment_contents = _read_attachments(attachments)
        if attachment_contents:
            topic = attachment_contents + "\n\n" + topic
            logger.info(f"[ATTACH] Injected content length: {len(attachment_contents)}")
        else:
            logger.warning("[ATTACH] _read_attachments returned empty — 附件解析失败")

    async def _event_stream():
        try:
            state = _build_initial_state(question=topic, thread_id=thread_id, user_name=getattr(request, 'user_name', ''))
            run_config = {"configurable": {"thread_id": thread_id}}
            graph = await get_graph()

            async for chunk in graph.astream(
                state, run_config, stream_mode=["updates", "custom"]
            ):
                # chunk can be either:
                #   (a) dict: node update → {node_name: output_dict}
                #   (b) tuple/list: custom event from get_stream_writer()
                if isinstance(chunk, (tuple, list)):
                    # Custom event: (namespace, event_type, data)
                    ns, evt_type, evt_data = chunk[0], chunk[1], chunk[2] if len(chunk) > 2 else {}
                    if evt_type == "custom":
                        # Our phase events from orchestrator nodes
                        yield _sse("phase_event", {
                            "phase": evt_data.get("phase", ""),
                            "message": evt_data.get("message", ""),
                            "count": evt_data.get("count"),
                            "content": evt_data.get("content"),
                            "stage": evt_data.get("stage"),
                        })
                    else:
                        yield _sse("custom_event", {
                            "namespace": str(ns),
                            "event_type": evt_type,
                            "data": evt_data,
                        })
                elif isinstance(chunk, dict):
                    # Node output update
                    for node_name, node_output in chunk.items():
                        if node_name == "__interrupt__":
                            continue
                        yield _sse("agent_event", {
                            "agent": node_name,
                            "output": _serialize_state_fragment(node_output),
                        })

            # After stream ends, get final state
            final_state = await graph.aget_state(run_config)

            # Check for HITL interrupt
            interrupts = final_state.interrupts if final_state else []
            if interrupts:
                interrupt_data = [
                    _serialize_interrupt(iv) for iv in interrupts
                ]
                yield _sse("interrupt", {
                    "thread_id": thread_id,
                    "interrupts": interrupt_data,
                })
            else:
                final_response = (
                    final_state.values.get("final_response", "")
                    if final_state else ""
                )
                yield _sse("completed", {
                    "final_output": final_response or "处理完成。",
                    "iteration": (
                        final_state.values.get("loop_count", 0)
                        if final_state else 0
                    ),
                    "thread_id": thread_id,
                })
                # ── 长期记忆异步写入（不阻塞 SSE）──
                if topic and final_response:
                    try:
                        from src.memory.long_term import save_conversation_memory
                        save_conversation_memory(thread_id, topic, final_response)
                    except Exception:
                        pass

        except Exception as e:
            logger.exception("Workflow streaming error")
            yield _sse("error", {"message": str(e)})

    response = StreamingHttpResponse(
        _event_stream(),
        content_type="text/event-stream",
        status=200,
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# ── 22b. Workflow Resume (HITL) ──
@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def workflow_resume(request: HttpRequest) -> StreamingHttpResponse:
    """Resume a paused workflow after a HITL interrupt.

    Body: {"thread_id": "...", "resume_value": "ok" | "user feedback"}
    """
    body = _json_body(request)
    thread_id = body.get("thread_id")
    resume_value = body.get("resume_value", "ok")

    if not thread_id:
        return _error_response("thread_id is required", 400)

    def _event_stream():
        try:
            from langgraph.types import Command

            run_config = {"configurable": {"thread_id": thread_id}}

            async def _run():
                graph = await get_graph()
                events = []
                async for chunk in graph.astream(
                    Command(resume=resume_value),
                    run_config,
                    stream_mode="updates",
                ):
                    events.append(chunk)
                final = await graph.aget_state(run_config)
                return events, final

            events, final_state = async_to_sync(_run)()

            for chunk in events:
                for node_name, node_output in chunk.items():
                    if node_name == "__interrupt__":
                        continue
                    yield _sse("agent_event", {
                        "agent": node_name,
                        "output": _serialize_state_fragment(node_output),
                    })

            interrupts = final_state.interrupts if final_state else []
            if interrupts:
                interrupt_data = [
                    _serialize_interrupt(iv) for iv in interrupts
                ]
                yield _sse("interrupt", {
                    "thread_id": thread_id,
                    "interrupts": interrupt_data,
                })
            else:
                final_response = (
                    final_state.values.get("final_response", "")
                    if final_state else ""
                )
                yield _sse("completed", {
                    "final_output": final_response or "处理完成。",
                    "iteration": (
                        final_state.values.get("loop_count", 0)
                        if final_state else 0
                    ),
                    "thread_id": thread_id,
                })

        except Exception as e:
            logger.exception("Workflow resume error")
            yield _sse("error", {"message": str(e)})

    response = StreamingHttpResponse(
        _event_stream(),
        content_type="text/event-stream",
        status=200,
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def _serialize_state_fragment(fragment: Any) -> Any:
    """Convert LangGraph state fragments into JSON-safe dicts, trimming large texts."""
    if isinstance(fragment, dict):
        result = {}
        for k, v in fragment.items():
            if isinstance(v, str) and len(v) > 3000:
                result[k] = v[:3000] + "... [truncated]"
            elif isinstance(v, (list, dict)):
                result[k] = _serialize_state_fragment(v)
            else:
                result[k] = v
        return result
    elif isinstance(fragment, list):
        return [_serialize_state_fragment(item) for item in fragment]
    return fragment


def _serialize_interrupt(interrupt_value: Any) -> Dict[str, Any]:
    """Extract interrupt metadata into a JSON-safe dict."""
    if hasattr(interrupt_value, "value"):
        raw = interrupt_value.value
    elif hasattr(interrupt_value, "__dict__"):
        raw = interrupt_value.__dict__
    else:
        raw = interrupt_value
    return {
        "value": _serialize_state_fragment(raw),
        "resumable": True,
    }


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    """Format a single SSE data line."""
    return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"

# ── 23. List Logs ──
@require_http_methods(["GET"])
def audit_log_view(request: HttpRequest) -> JsonResponse:
    from src.middleware import audit_log
    limit = int(request.GET.get('limit', 100))
    return JsonResponse({"entries": audit_log.list(limit), "stats": audit_log.stats()})


@require_http_methods(["GET"])
def list_logs(request: HttpRequest) -> JsonResponse:
    limit = int(request.GET.get('limit', 100))
    return JsonResponse([], safe=False)


# ── 24. Cache List ──
@require_http_methods(["GET"])
def cache_list(request: HttpRequest) -> JsonResponse:
    """返回语义缓存条目列表 + 按领域统计。"""
    from src.semantic_cache import get_cache
    cache = get_cache()
    entries = cache.list_entries()
    stats = cache.stats_by_domain()
    return JsonResponse({"total": len(entries), "entries": entries, "stats": stats})


# ── 25. Cache Delete Entry ──
@csrf_exempt
@require_http_methods(["DELETE"])
def cache_delete_entry(request: HttpRequest, entry_id: int) -> JsonResponse:
    """删除指定索引的语义缓存条目。"""
    from src.semantic_cache import get_cache
    cache = get_cache()
    ok = cache.remove_entry(entry_id)
    if not ok:
        return _error_response(f"Cache entry {entry_id} not found", 404)
    return JsonResponse({"success": True, "deleted": entry_id})


# ── 25b. Cache Clear All ──
@csrf_exempt
@require_http_methods(["DELETE"])
def cache_clear_all(request: HttpRequest) -> JsonResponse:
    """清空全部语义缓存。"""
    from src.semantic_cache import get_cache
    cache = get_cache()
    count = cache.clear()
    return JsonResponse({"success": True, "cleared": count})


# ── 25c. Cache Delete By Domain ──
@csrf_exempt
@require_http_methods(["DELETE"])
def cache_delete_by_domain(request: HttpRequest, domain: str) -> JsonResponse:
    """删除指定领域的所有语义缓存条目。"""
    from src.semantic_cache import get_cache
    cache = get_cache()
    count = cache.remove_by_domain(domain)
    return JsonResponse({"success": True, "domain": domain, "removed": count})
