"""Research Agent: Information gathering with real search tools.

Uses:
- knowledge_search: FAISS + BM25 hybrid local index
- web_search: Baidu AI search → Tavily fallback
- LLM synthesis: condenses retrieved chunks into concise, readable answers
"""

import asyncio
import json
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.core.goal_manager import GoalPriority
from src.core.exceptions import (
    AgentExecutionError,
    SearchError,
    ToolExecutionError,
)
from src.core.logging_config import get_logger
from src.tools import knowledge_search, web_search
from src.resilience import retry

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """Agent specialized in research and information gathering.

    Responsibilities:
    - Search internal knowledge base (FAISS + BM25)
    - Search external web (Baidu → Tavily)
    - Synthesize concise answer from retrieved content via LLM
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        message_bus=None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type="research",
            message_bus=message_bus,
        )

    async def deliberate(self, state: Dict[str, Any]) -> None:
        """Set research-oriented goals."""
        question = self._get_query_text(state)
        self.goal_manager.create_goal(
            description=f"Research: gather information about '{question[:80]}'",
            priority=GoalPriority.HIGH,
            metadata={"type": "research", "query": question},
        )

    async def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research: search internal KB → web search → LLM synthesis."""
        trace_id = state.get("trace_id", "")
        question = self._get_query_text(state)
        if not question:
            return {**state, "research_report": "No query provided.", "error": "empty_query"}

        try:
            # Phase 0: use pre-retrieved context from orchestrator if available
            pre_contexts = state.get("retrieved_context", []) or []
            pre_texts = [c.get("text", "") for c in pre_contexts if c.get("text")]

            # Phase 1: Internal knowledge base search (skip if pre-context covers it)
            kb_texts = list(pre_texts)  # start with orchestrator's context
            if not kb_texts:
                user_name = state.get("user_name", "")
                kb_texts = await self._search_internal_texts(question, trace_id, user_name=user_name)

            # Phase 2: External web search
            web_texts = self._search_external_texts(question, trace_id)

            # Extract conversation history from state.messages
            raw_messages = state.get("messages", []) or []
            history_messages: List[Dict] = []
            for m in raw_messages:
                role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
                history_messages.append({"role": role, "content": getattr(m, "content", "")})

            long_term_context = state.get("long_term_context", "") or ""

            # Phase 3: LLM synthesis of all retrieved content
            report = await self._synthesize(
                question, kb_texts, web_texts, trace_id, history_messages, long_term_context,
            )

            logger.info(
                "ResearchAgent.act: completed",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
            )

            return {
                **state,
                "research_report": report,
                "retrieved_context": [],
            }
        except Exception as e:
            logger.error(
                f"ResearchAgent.act failed: {e}",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                exc_info=True,
            )
            raise AgentExecutionError(
                f"Research agent execution failed for query '{question[:60]}': {e}",
                detail={
                    "agent_id": self.agent_id,
                    "query": question[:200],
                    "traceback": traceback.format_exc(),
                },
            ) from e

    async def _agentic_search(
        self, state: Dict[str, Any],
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Agentic path: LLM autonomously decides which tools to call, how many times.

        Uses LangGraph's create_react_agent with bind_tools for a lightweight
        agent loop. The LLM sees kb_search_tool, web_search_tool, calculate_tool
        and decides its own search strategy — no hardcoded phases.

        Args:
            system_prompt_override: If provided, replaces the default
                system_instruction. Used for domain-specific review tasks.
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from src.config import config
        from src.tools import LANGCHAIN_TOOLS

        trace_id = state.get("trace_id", "")
        question = self._get_query_text(state)

        pre_contexts = state.get("retrieved_context", []) or []
        kb_hint = ""
        if pre_contexts:
            snippets = [c.get("text", "")[:120] for c in pre_contexts[:2] if c.get("text")]
            if snippets:
                kb_hint = "\n".join(f"- {s}" for s in snippets)

        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.3,
            timeout=30,
        )

        agent = create_react_agent(
            model=llm.bind_tools(LANGCHAIN_TOOLS),
            tools=LANGCHAIN_TOOLS,
        )

        system_instruction = system_prompt_override or (
            "你是一个信息检索助手。请使用以下工具查找答案：\n"
            "- kb_search_tool: 搜索本地知识库（优先使用）\n"
            "- web_search_tool: 联网搜索最新信息\n"
            "- calculate_tool: 计算数学表达式\n\n"
            "策略：先搜知识库，结果不够再联网。\n\n"
            "【来源标注规则 — 必须准确标注实际来源】\n"
            "回答时必须标注内容来源：\n"
            "📄 文档来源 — 仅当内容来自 kb_search_tool 检索到的知识库原文时使用\n"
            "🌐 联网搜索 — 仅当内容来自 web_search_tool 的网络搜索结果时使用\n"
            "禁止将网络搜索的内容标注为「📄 文档来源」\n"
            "如果知识库有相关内容，以「📄 文档来源」开头输出原文，末尾加：\n"
            "「📌 以上为知识库已有内容。如需联网搜索或AI补充，请回复\"需要补充\"。」\n"
            "如果知识库无内容但网络搜索有结果，以「🌐 联网搜索」开头输出。"
        )

        user_content = f"问题：{question}"
        if kb_hint:
            user_content += f"\n\n知识库预检索线索（用于确定是否需要进一步搜索）：\n{kb_hint}"

        # ── 审查提取上下文注入（确定性前置提取结果） ──
        review_extraction_context = state.get("review_extraction_context", "") or ""
        if review_extraction_context:
            user_content = review_extraction_context + "\n\n" + user_content

        # ── 对话历史注入 ──
        history_messages = []
        raw_messages = state.get("messages", []) or []
        for m in raw_messages:
            role_label = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
            history_messages.append({"role": role_label, "content": getattr(m, "content", "")})
        if history_messages:
            history_text = "\n".join(
                f"{'用户' if h['role'] == 'user' else 'AI'}: {h['content']}"
                for h in history_messages if h["content"]
            )
            if history_text:
                user_content = f"对话历史：\n{history_text}\n\n{user_content}"

        # ── 长期记忆注入 ──
        long_term_context = state.get("long_term_context", "") or ""
        if long_term_context:
            user_content = f"跨会话长期记忆（之前相关对话的摘要，可参考以更好地理解当前问题）：\n{long_term_context}\n\n{user_content}"

        try:
            result = await agent.ainvoke(
                {"messages": [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_content),
                ]},
                config={"recursion_limit": 10},
            )
            messages = result.get("messages", [])
            final = messages[-1].content if messages else "未能生成回答。"

            logger.info(
                f"ResearchAgent._agentic_search: completed, {len(messages)} messages",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
            )

            return {
                **state,
                "research_report": final,
                "retrieved_context": [],
            }
        except Exception as e:
            logger.error(
                f"ResearchAgent._agentic_search failed: {e}",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                exc_info=True,
            )
            # Fallback to fast path
            logger.warning(
                "Falling back to fast path after agentic failure",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
            )
            return await self.act(state)

    async def _search_internal_texts(self, question: str, trace_id: str = "", user_name: str = "", return_docs: bool = False):
        """Search internal KB and return raw chunk texts.

        Args:
            return_docs: If True, returns (texts, doc_basenames) tuple.
                         doc_basenames is the list of referenced document names (for cache permission filtering).
        """
        texts: List[str] = []
        doc_names: List[str] = []
        try:
            raw = await asyncio.to_thread(
                knowledge_search, question, top_k=10, user_name=user_name
            )
            result = json.loads(raw)
            # knowledge_search 已通过 user_name 参数完成权限过滤，无需二次过滤

            if not result.get("found") or not result.get("results"):
                logger.info(
                    "ResearchAgent: internal search returned no results",
                    extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                )
                return (texts, doc_names) if return_docs else texts

            seen_docs = set()
            for r in result.get("results", []):
                text = r.get("text", "")
                if text:
                    texts.append(text)
                doc_id = r.get("doc_id", "")
                if doc_id and "::" in doc_id:
                    basename = doc_id.split("::")[0]
                    if basename not in seen_docs:
                        seen_docs.add(basename)
                        doc_names.append(basename)

            logger.info(
                f"ResearchAgent: internal search found {len(texts)} chunks, {len(doc_names)} docs",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
            )
        except SearchError:
            raise
        except Exception as e:
            logger.error(
                f"Internal search failed: {e}",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                exc_info=True,
            )
            raise SearchError(
                f"Internal knowledge base search failed: {e}",
                detail={"agent_id": self.agent_id, "query": question[:200]},
            ) from e

        return (texts, doc_names) if return_docs else texts

    def _search_external_texts(self, question: str, trace_id: str = "") -> List[str]:
        """Search external web and return snippet texts."""
        texts: List[str] = []
        try:
            raw = web_search(question, max_results=5)
            results = json.loads(raw)

            if not results:
                logger.info(
                    "ResearchAgent: external search returned no results",
                    extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                )
                return texts

            for r in results:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                if title or snippet:
                    texts.append(f"{title}: {snippet}" if title else snippet)

            logger.info(
                f"ResearchAgent: external search found {len(texts)} results",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
            )
        except SearchError:
            raise
        except Exception as e:
            logger.error(
                f"External search failed: {e}",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                exc_info=True,
            )
            raise SearchError(
                f"External web search failed: {e}",
                detail={"agent_id": self.agent_id, "query": question[:200]},
            ) from e

        return texts

    async def _synthesize(
        self, question: str, kb_texts: List[str], web_texts: List[str], trace_id: str = "",
        history_messages: Optional[List[Dict]] = None,
        long_term_context: str = "",
        system_prompt_override: Optional[str] = None,
    ) -> str:
        """Use LLM to synthesize retrieved texts into a structured answer.

        The LLM is instructed to:
        - Preserve the heading hierarchy from the source document
        - Output retrieved content in full under each heading, do NOT paraphrase or abbreviate
        - Keep the answer clear and easy to understand

        Args:
            system_prompt_override: If provided, use this as the system prompt
                instead of the default. Used for domain-specific review tasks.
        """
        # Build context from retrieved texts
        context_parts = []

        if kb_texts:
            context_parts.append("【内部知识库内容】")
            for i, t in enumerate(kb_texts, 1):
                context_parts.append(f"[{i}] {t}")
        # KB 为空时不显示 KB 区域，避免 LLM 误标「📄 文档来源」

        if web_texts:
            context_parts.append("\n【外部网络搜索结果】")
            for i, t in enumerate(web_texts[:3], 1):
                context_parts.append(f"[{i}] {t}")

        context = "\n\n".join(context_parts)

        # Build the LLM prompt
        system_prompt = system_prompt_override or (
            "你是一个知识整理助手。请根据提供的参考资料回答用户的问题。\n"
            "要求：\n"
            "1. 严格基于参考资料回答，不要编造信息\n"
            "2. 按原文的标题层级组织答案：\n"
            "   - 如果用户询问某个三级标题，列出该标题下的四级、五级标题及其完整原文内容\n"
            "   - 如果用户询问某个二级标题，列出该标题下的三级、四级标题及其完整原文内容\n"
            "   - 以此类推，按标题层级归类输出\n"
            "3. 已检索到的原文内容需完整输出，不要精简、省略或用自己的话改写\n"
            "4. 如果参考资料中完全没有答案，诚实告知\n"
            "5. 【来源标注规则 — 必须准确标注实际来源】\n"
            "   根据参考资料的实际来源使用以下标记：\n"
            "   📄 文档来源 — 仅当参考资料中包含【内部知识库内容】区域且有实际文本时使用此标记\n"
            "   🌐 联网搜索 — 当参考资料中只有【外部网络搜索结果】区域时必须使用此标记\n"
            "   ⚠️ 判断规则：如果参考资料中没有【内部知识库内容】区域，说明知识库无相关内容，必须使用「🌐 联网搜索」\n"
            "   如果内部知识库有内容，以「📄 文档来源」开头输出知识库原文\n"
            "   如果内部知识库无内容但网络搜索有结果，以「🌐 联网搜索」开头输出网络搜索结果\n"
            "   禁止将网络搜索的内容标注为「📄 文档来源」\n"
            "6. 【分层输出规则】\n"
            "   第一层（默认输出）：输出检索到的内容（知识库优先）。\n"
            "   - 如果参考资料包含【内部知识库内容】：按标题层级整理输出，末尾追加：\n"
            "     「📌 以上为知识库已有内容。如需联网搜索最新信息或 AI 补充通用知识，请回复\"需要补充\"。」\n"
            "   - 如果参考资料不包含【内部知识库内容】但有【外部网络搜索结果】：直接输出网络搜索结果，末尾追加：\n"
            "     「📌 以上为网络搜索结果。如需 AI 补充通用知识，请回复\"需要补充\"。」\n"
            "   第二层（仅在用户明确要求\"补充\"/\"继续\"/\"更多\"后触发）：\n"
            "   - 输出其他来源的补充内容\n"
            "7. 【无附件兜底框架规则 — 必须遵守】\n"
            "   当用户要求审查/分析/检查某类文档但参考资料中没有该文档的具体文本时：\n"
            "     a. 坦诚说明未收到具体文档（一句话即可）\n"
            "     b. 基于知识库立即给出一份完整的通用审查框架\n"
            "     c. 框架末尾引导用户上传具体文件以获得针对性深度分析\n"
            "     d. 禁止仅回复「未提供文档，无法分析」就结束"
        )

        # ── 对话历史注入 ──
        history_text = ""
        if history_messages:
            history_lines = []
            for m in history_messages:
                role = "用户" if m.get("role") == "user" else "AI"
                content = m.get("content", "")
                if content:
                    history_lines.append(f"{role}: {content}")
            if history_lines:
                history_text = "\n".join(history_lines)

        user_prompt = f"用户问题：{question}\n\n"
        if history_text:
            user_prompt += f"对话历史（当前问题之前的上下文）：\n{history_text}\n\n"
        if long_term_context:
            user_prompt += f"跨会话长期记忆（之前相关对话的摘要）：\n{long_term_context}\n\n"
        user_prompt += f"参考资料：\n{context}\n\n请按标题层级归类，完整输出检索到的内容。"

        try:
            content = await self._llm_invoke(system_prompt, user_prompt)

            logger.info(
                f"ResearchAgent._synthesize: LLM generated {len(content)} chars",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
            )
            return content

        except Exception as e:
            logger.error(
                f"ResearchAgent._synthesize LLM call failed: {e}",
                extra={"component": "research_agent", "agent_id": self.agent_id, "trace_id": trace_id},
                exc_info=True,
            )
            # Fallback: return a minimal summary without LLM
            return self._fallback_synthesize(question, kb_texts, web_texts)

    @retry(max_retries=2, base_delay=1.0, max_delay=8.0)
    async def _llm_invoke(self, system_prompt: str, user_prompt: str) -> str:
        """Invoke LLM with retry support. Extracted for retry decoration."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from src.config import config

        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.3,
            max_tokens=8192,
            timeout=30,
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = await llm.ainvoke(messages)
        result = response.content.strip() if hasattr(response, "content") else str(response).strip()
        try:
            from src.middleware import token_tracker
            meta = getattr(response, 'response_metadata', {}) or {}
            usage = meta.get('token_usage', {})
            inp = usage.get('prompt_tokens', 0)
            out = usage.get('completion_tokens', 0)
            if inp or out:
                token_tracker.record(inp, out, module="research_agent")
        except Exception:
            pass
        return result

    async def _synthesize_stream(
        self, question: str, kb_texts: List[str], web_texts: List[str], trace_id: str = "",
        history_messages: Optional[List[Dict]] = None,
        long_term_context: str = "",
        system_prompt_override: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Streaming version of _synthesize. Yields tokens as the LLM generates them.

        Args:
            system_prompt_override: If provided, use this as the system prompt
                instead of the default. Used for domain-specific review tasks
                (e.g., credit report review with REVIEW_SYSTEM_PROMPT).
        """
        context_parts = []
        if kb_texts:
            context_parts.append("【内部知识库内容】")
            for i, t in enumerate(kb_texts, 1):
                context_parts.append(f"[{i}] {t}")
        # KB 为空时不显示 KB 区域，避免 LLM 误标「📄 文档来源」
        if web_texts:
            context_parts.append("\n【外部网络搜索结果】")
            for i, t in enumerate(web_texts[:3], 1):
                context_parts.append(f"[{i}] {t}")
        context = "\n\n".join(context_parts)

        system_prompt = system_prompt_override or (
            "你是一个知识整理助手。请根据提供的参考资料回答用户的问题。\n"
            "要求：\n"
            "1. 严格基于参考资料回答，不要编造信息\n"
            "2. 按原文的标题层级组织答案\n"
            "3. 已检索到的原文内容需完整输出，不要精简、省略或用自己的话改写\n"
            "4. 如果参考资料中完全没有答案，诚实告知\n"
            "5. 【来源标注规则 — 必须准确标注实际来源】\n"
            "   📄 文档来源 — 仅当参考资料中包含【内部知识库内容】区域且有实际文本时使用\n"
            "   🌐 联网搜索 — 当参考资料中只有【外部网络搜索结果】区域时必须使用此标记\n"
            "   ⚠️ 判断规则：如果参考资料中没有【内部知识库内容】区域，说明知识库无相关内容，必须使用「🌐 联网搜索」\n"
            "   禁止将网络搜索的内容标注为「📄 文档来源」\n"
            "6. 【分层输出】默认输出检索到的内容（知识库优先）。\n"
            "   - 参考资料包含【内部知识库内容】时，以「📄 文档来源」开头，末尾加：\n"
            "     「📌 以上为知识库已有内容。如需联网搜索或AI补充，请回复\"需要补充\"。」\n"
            "   - 参考资料不包含【内部知识库内容】但包含【外部网络搜索结果】时，以「🌐 联网搜索」开头，末尾加：\n"
            "     「📌 以上为网络搜索结果。如需AI补充，请回复\"需要补充\"。」\n"
            "   仅当用户明确要求后再输出其他来源的补充内容。\n"
            "7. 当用户要求审查/分析某类文档但参考资料中没有该文档具体文本时，\n"
            "   一句话说明未收到文档，然后基于知识库立刻给出通用审查框架，\n"
            "   并在末尾引导用户上传文件。禁止仅回复「未提供文档」就结束。\n"
        )

        history_text = ""
        if history_messages:
            history_lines = []
            for m in history_messages:
                role = "用户" if m.get("role") == "user" else "AI"
                content = m.get("content", "")
                if content:
                    history_lines.append(f"{role}: {content}")
            if history_lines:
                history_text = "\n".join(history_lines)

        user_prompt = f"用户问题：{question}\n\n"
        if history_text:
            user_prompt += f"对话历史（当前问题之前的上下文）：\n{history_text}\n\n"
        if long_term_context:
            user_prompt += f"跨会话长期记忆（之前相关对话的摘要）：\n{long_term_context}\n\n"
        user_prompt += f"参考资料：\n{context}\n\n请按标题层级归类，完整输出检索到的内容。"

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from src.config import config

        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.3,
            max_tokens=8192,
            timeout=30,
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def _fallback_synthesize(
        self, question: str, kb_texts: List[str], web_texts: List[str]
    ) -> str:
        """Minimal fallback when LLM is unavailable."""
        parts = [f"# 查询结果\n\n**问题**: {question}\n"]
        if kb_texts:
            parts.append(f"**知识库匹配**: {len(kb_texts)} 条相关片段\n")
            parts.append(kb_texts[0][:500])
        elif web_texts:
            parts.append(f"**网络搜索**: {len(web_texts)} 条结果\n")
            parts.append(web_texts[0][:500])
        else:
            parts.append("未找到相关内容。")
        return "\n".join(parts)

    def _get_query_text(self, state: Dict[str, Any]) -> str:
        """Extract query from state."""
        return (
            state.get("question", "")
            or state.get("raw_input", "")
            or state.get("task_description", "")
        )
