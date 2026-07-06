"""工具函数模块 — 提供 Agent 可调用的搜索 / 计算 / 文件操作。

搜索工具支持多后端：
  1. 百度 AI 搜索（默认，需 BAIDU_API_KEY）
  2. Tavily（备用，需 TAVILY_API_KEY）
  3. 本地知识库 FAISS + BM25 混合检索

熔断保护：百度和 Tavily 各自独立熔断。
"""

import json
import logging
import os
import hashlib
from typing import Optional

from .resilience import baidu_circuit, tavily_circuit, CircuitBreakerOpenError
from .core.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD = 0.3


# ── 联网搜索 ─────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> str:
    """网络搜索工具。百度 AI 搜索，失败回退 Tavily。

    Returns:
        JSON 格式搜索结果字符串。
    """
    logger.info(
        "web_search started",
        extra={"component": "tools", "query": query[:100], "max_results": max_results},
    )
    try:
        results = _baidu_search(query, max_results)
        if not results:
            logger.warning(
                "web_search: no results",
                extra={"component": "tools", "query": query[:100]},
            )
            return json.dumps([], ensure_ascii=False)
        results = _rerank_web_results(query, results, top_k=max_results)
        logger.info(
            f"web_search: returned {len(results)} results",
            extra={"component": "tools", "query": query[:100]},
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception:
        logger.error(
            f"web_search failed for query: {query[:100]}",
            extra={"component": "tools", "query": query[:100]},
            exc_info=True,
        )
        raise


def _rerank_web_results(query: str, results: list[dict], threshold: float = THRESHOLD, top_k: int = 5) -> list[dict]:
    """CrossEncoder 精排过滤联网搜索结果。"""
    if len(results) <= 1:
        return results
    try:
        from .rag.reranker import Reranker
        reranker = Reranker()
        if not reranker.available:
            logger.warning(
                "Reranker unavailable, skipping rerank",
                extra={"component": "tools"},
            )
            return results[:top_k]
        candidates = []
        for r in results:
            text = f"{r['title']} {r['snippet']}".strip()
            candidates.append((r["title"], text, 0.0))
        reranked = reranker.rerank(query, candidates, top_k=top_k)
        filtered = []
        for title, _text, score in reranked:
            if score < threshold:
                continue
            for r in results:
                if r["title"] == title:
                    out = dict(r)
                    out["rerank_score"] = round(score, 4)
                    filtered.append(out)
                    break
        logger.info(
            f"CrossEncoder filtered: {len(results)} → {len(filtered)}",
            extra={"component": "tools"},
        )
        return filtered
    except Exception as e:
        logger.warning(
            f"CrossEncoder rerank exception (degraded): {e}",
            extra={"component": "tools"},
        )
        return results[:top_k]


def _baidu_search(query: str, max_results: int) -> list[dict]:
    """百度 AI 搜索。"""
    api_key = os.getenv("BAIDU_API_KEY")
    if not api_key:
        logger.warning(
            "BAIDU_API_KEY not configured, falling back to Tavily",
            extra={"component": "tools"},
        )
        return _try_tavily_fallback(query, max_results)
    if baidu_circuit.is_open():
        logger.warning(
            "[BaiduSearch] circuit breaker open, falling back to Tavily",
            extra={"component": "tools"},
        )
        return _try_tavily_fallback(query, max_results)

    def _do_request():
        import httpx
        response = httpx.post(
            "https://qianfan.baidubce.com/v2/ai_search/web_search",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "messages": [{"content": query, "role": "user"}],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web", "top_k": max_results}],
            },
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {"title": item.get("title", ""), "snippet": item.get("content", ""), "source": item.get("url", "")}
            for item in data.get("references", [])[:max_results]
        ]

    try:
        results = baidu_circuit.call_blocking(_do_request)
        if not results:
            return _try_tavily_fallback(query, max_results)
        return results
    except CircuitBreakerOpenError:
        return _try_tavily_fallback(query, max_results)
    except Exception as e:
        logger.error(
            f"Baidu search failed: {e}",
            extra={"component": "tools", "query": query[:100]},
        )
        return _try_tavily_fallback(query, max_results)


def _try_tavily_fallback(query: str, max_results: int) -> list[dict]:
    """Tavily 备用搜索。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning(
            "TAVILY_API_KEY not configured",
            extra={"component": "tools"},
        )
        return []
    if tavily_circuit.is_open():
        logger.warning(
            "[TavilySearch] circuit breaker open",
            extra={"component": "tools"},
        )
        return []

    def _do_request():
        import httpx
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results, "search_depth": "basic"},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {"title": r.get("title", ""), "snippet": r.get("content", ""), "source": r.get("url", "")}
            for r in data.get("results", [])
        ]

    try:
        return tavily_circuit.call_blocking(_do_request)
    except CircuitBreakerOpenError:
        return []
    except Exception as e:
        logger.error(
            f"Tavily search failed: {e}",
            extra={"component": "tools", "query": query[:100]},
        )
        return []


# ── 查询扩展 ─────────────────────────────────────────────

from cachetools import TTLCache
_expand_query_cache = TTLCache(maxsize=128, ttl=300)


def expand_query(query: str) -> list:
    """LLM Query Expansion：生成 2 条改写查询，提升检索召回率。"""
    cache_key = query.strip().lower()
    if cache_key in _expand_query_cache:
        return _expand_query_cache[cache_key]
    try:
        from .config import config
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.2,
            max_tokens=256,
            timeout=10,
        )
        prompt = f"""请将以下用户问题改写成 2 个不同角度的搜索查询。
要求：保持原意，但使用不同的表述方式。改写 1：更正式的书面语。改写 2：更简洁的关键词式。每行一个，不要序号。
原始问题：{query}"""
        resp = llm.invoke([HumanMessage(content=prompt)])
        lines = [l.strip() for l in resp.content.strip().split("\n") if l.strip()]
        expanded = [query] + lines[:2]
        logger.info(
            f"Query Expansion: {query[:60]} → {len(expanded)} queries",
            extra={"component": "tools"},
        )
        _expand_query_cache[cache_key] = expanded
        return expanded
    except Exception as e:
        logger.warning(
            f"Query Expansion failed (degraded): {e}",
            extra={"component": "tools"},
        )
        return [query]


# ── 知识库搜索 ───────────────────────────────────────────

def knowledge_search(query: str, top_k: int = 5, min_rrf_score: float = 0.005, domain: Optional[str] = None) -> str:
    """本地 FAISS + BM25 + RRF 混合索引知识库搜索。

    Args:
        query: 搜索查询文本。
        top_k: 返回结果数量。
        min_rrf_score: 最低相关度阈值。
        domain: 领域过滤标签（"finance"/"contract"/"law"/"general"），
                None 或 "general" 时不过滤。

    Returns:
        JSON 格式搜索结果。
    """
    logger.info(
        "knowledge_search started",
        extra={"component": "tools", "query": query[:100], "top_k": top_k},
    )
    try:
        from .rag.indexer import get_indexer

        uploads_dir = os.path.join(PROJECT_ROOT, "uploads")
        indexes_dir = os.path.join(PROJECT_ROOT, "indexes")

        indexer = get_indexer(uploads_dir=uploads_dir, indexes_dir=indexes_dir)

        queries = expand_query(query)
        search_results = [indexer.search(q, top_k=top_k, domain=domain) for q in queries]

        all_results, seen_docs = [], set()
        total_bm25, total_dense = 0, 0
        for sr in search_results:
            for item in sr.get("results", []):
                # Content-hash dedup: same text = same chunk, skip duplicates
                chunk_text = item.get("chunk", "")
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                if chunk_hash in seen_docs:
                    continue
                seen_docs.add(chunk_hash)
                all_results.append(item)
            total_bm25 += sr.get("stats", {}).get("bm25_hits", 0)
            total_dense += sr.get("stats", {}).get("dense_hits", 0)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = all_results[:top_k]

        if not results:
            return json.dumps({
                "source": "internal_knowledge_base",
                "found": False,
                "query": query,
                "total": 0,
                "results": [],
                "note": "知识库中未找到匹配内容，建议使用外部搜索。",
            }, ensure_ascii=False, indent=2)

        max_score = max(r.get("score", 0) for r in results)
        found = max_score >= min_rrf_score

        formatted = {
            "source": "internal_knowledge_base",
            "found": found,
            "query": query,
            "total": len(results),
            "max_score": round(max_score, 6),
            "total_bm25": total_bm25,
            "total_dense": total_dense,
            "results": [
                {
                    "doc_id": r.get("doc_id", ""),
                    "text": r.get("chunk", ""),
                    "score": round(r.get("score", 0), 6),
                }
                for r in results
            ],
            "note": "来自内部知识库的匹配结果。" if found else "匹配度较低，建议结合外部搜索补充。",
        }
        logger.info(
            f"knowledge_search: found {len(results)} results",
            extra={"component": "tools", "query": query[:100]},
        )
        return json.dumps(formatted, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(
            f"knowledge_search exception: {e}",
            extra={"component": "tools", "query": query[:100]},
            exc_info=True,
        )
        raise


# ── 工具函数 ─────────────────────────────────────────────

def calculate(expression: str) -> str:
    """安全计算数学表达式。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def read_local_file(file_path: str, encoding: str = "utf-8") -> str:
    """读取本地文件内容。"""
    if not os.path.exists(file_path):
        return f"文件不存在: {file_path}"
    try:
        with open(file_path, "r", encoding=encoding) as f:
            content = f.read()
        return content[:5000]
    except Exception as e:
        return f"读取失败: {e}"


TOOL_REGISTRY = {
    "web_search": web_search,
    "knowledge_search": knowledge_search,
    "calculate": calculate,
    "read_local_file": read_local_file,
}


def get_tool(name: str) -> Optional[callable]:
    return TOOL_REGISTRY.get(name)


def list_tools() -> list[str]:
    return list(TOOL_REGISTRY.keys())


# ── LangChain Tool Wrappers (for native Tool Calling / bind_tools) ──

from langchain_core.tools import tool


@tool
def kb_search_tool(query: str, domain: Optional[str] = None) -> str:
    """搜索本地知识库（FAISS + BM25 混合索引）。

    当用户询问已上传文档中的内容时优先使用此工具。
    返回与查询最相关的文档片段及其相关性分数。

    Args:
        query: 搜索查询词或问题
        domain: 可选领域标签 "finance"/"contract"/"law"，限定搜索范围。
    """
    raw = knowledge_search(query, top_k=5, domain=domain)
    result = json.loads(raw)
    if not result.get("found") or not result.get("results"):
        return "知识库中未找到相关内容。"
    parts = []
    for i, r in enumerate(result.get("results", []), 1):
        text = r.get("text", "")
        score = r.get("score", 0)
        parts.append(f"[{i}] (相关性: {score:.2f})\n{text[:600]}")
    return "\n\n".join(parts)


@tool
def web_search_tool(query: str) -> str:
    """联网搜索最新信息（百度 AI 搜索 → Tavily 备用）。

    当用户询问实时信息、新闻、天气、或知识库中没有的最新内容时使用此工具。

    Args:
        query: 搜索关键词或问题
    """
    raw = web_search(query, max_results=5)
    results = json.loads(raw)
    if not results:
        return "未找到相关网络搜索结果。"
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        source = r.get("source", "")
        parts.append(f"[{i}] {title}\n{snippet[:400]}\n来源: {source}")
    return "\n\n".join(parts)


@tool
def calculate_tool(expression: str) -> str:
    """安全计算数学表达式。

    支持基本算术运算：+、-、*、/、**（幂）、% （取余）。
    不支持导入模块、函数调用等危险操作。

    Args:
        expression: 数学表达式字符串，如 "2 + 3 * 4" 或 "(100 - 20) / 4"
    """
    return calculate(expression)


# 供 bind_tools() 使用的工具列表
LANGCHAIN_TOOLS = [kb_search_tool, web_search_tool, calculate_tool]
