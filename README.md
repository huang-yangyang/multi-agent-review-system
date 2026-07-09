1. 项目概述
本系统是一个面向文档审查与知识问答的多智能体（Multi-Agent）系统，核心能力如下：

BDI 认知架构：每个 Agent 内置 信念（Belief）、愿望（Desire） 和 意图（Intention） 三大模块，形成 感知 → 慎思 → 计划 → 行动 的认知闭环。

LangGraph 工作流引擎：使用有状态图（StateGraph）编排多 Agent 协作，支持条件路由、检查点（Checkpoint）持久化和流式 Token 输出。

领域自适应：自动识别 finance（授信/风控）、contract（合同）、law（法律）、general（通用）领域，审查类问题走确定性 Map-Reduce 管线。

混合检索（RAG）：结合 BM25 关键词检索与 Qdrant 向量稠密检索，通过 RRF 融合、CrossEncoder 精排和语义缓存，提升检索质量。

权限隔离：基于角色的文档可见性模型，在检索层做预过滤（Pre-filter），在缓存层做二次校验。

技术栈概览

层级	选型
工作流编排	LangGraph（StateGraph + AsyncSqliteSaver 检查点）
Agent 框架	自研 BDI 基类 + LangChain create_react_agent（用于复杂路径）
LLM	OpenAI / DeepSeek / 本地模型（可切换，带熔断与 Fallback）
向量检索	Qdrant（本地磁盘模式）+ BM25 关键词检索
认知知识库	ChromaDB（向量）+ SQLite（兜底）
状态/通信	Redis（可选，自动降级为内存）+ 内存异步消息总线
Web 服务	Django + ASGI（Uvicorn）
前端	Vue 3 + Vite
2. BDI 认知架构
BDI（Belief-Desire-Intention）是 Agent 认知建模的经典范式。本系统将其落地为三个核心模块，由 BaseAgent 串成认知循环。

BDI 要素	对应模块	职责
信念（Belief）	src/core/belief_base.py	知识/感知存储与检索（ChromaDB 向量 + SQLite 兜底），支持 TTL 和分类。
愿望（Desire）	src/core/goal_manager.py	目标生命周期管理，包括优先级队列、状态机和层级分解。
意图（Intention）	src/core/plan_executor.py	将目标分解为步骤，管理依赖图，并分步执行与跟踪进度。
2.1 认知循环
所有 Agent 继承 BaseAgent，其 run() 方法实现标准闭环：

感知（Perceive） → 更新 BeliefBase（观察环境、记录用户输入）。

慎思（Deliberate） → 基于 Belief 创建目标（GoalManager）。

计划（Plan） → 为激活的目标生成执行计划（PlanExecutor）。

行动（Act） → 执行计划步骤，产出结果（由子类实现）。







2.2 核心模块详解
BeliefBase (信念库)

双后端：ChromaDB（语义向量检索，优先） + SQLite（关键词兜底，LIKE 匹配）。

数据管理：数据带有 category 标签（如 perception、message、analysis_report），支持分类过滤和 TTL 自动过期（默认 86400 秒）。

API：add_knowledge / query / get_context / get_by_id / list_by_category / delete / clear_category / count。

GoalManager (目标管理)

Goal 数据类：包含 priority（高/中/低）、status（pending → active → completed/failed）、parent_goal_id 和 sub_goals（支持层级分解）。

调度逻辑：get_next_goal() 按优先级取最高待处理目标，且父子目标完成状态自动联动。

容量限制：max_goals 默认 20，超限拒绝新建。

PlanExecutor (计划执行)

Plan/PlanStep：步骤支持 dependencies（依赖其他步骤）。

生成与执行：generate_plan() 支持显式步骤或启发式拆分；execute_step() / get_ready_steps() 支持依赖满足的步骤并行执行。

进度跟踪：get_progress() 返回完成百分比与状态统计。

MessageBus (Agent 间通信)

提供四种完全异步的通信模式：P2P、PubSub、Request-Response、Broadcast。

消息信封（Message）支持 HMAC 签名（verify_signature）保证完整性，默认使用内存交付。

StateManager (分布式状态)

后端：Redis（可用时）+ 内存兜底（自动降级）。

并发控制：支持基于版本号的 CAS（compare_and_set）乐观锁。

操作：支持 TTL、批量操作（mget/mset）、命名空间操作（get_by_prefix/delete_by_prefix）和 JSON 序列化。

3. 系统总体架构
系统采用分层设计，职责清晰：

接入层：Django 提供 REST 端点，承载鉴权、会话、文件上传和流式 SSE 响应。

编排层：LangGraph Orchestrator 负责任务分解、知识预检索、路由与结果聚合。

Agent 层：专业化 Agent 执行检索、分析和审查等任务，内部遵循 BDI 认知闭环。

认知核心层：BDI 三件套、消息总线和状态管理，供所有 Agent 复用。

数据与检索层：负责混合 RAG 索引、文档库、语义缓存和权限模型。

4. LangGraph 工作流编排引擎
src/workflows/orchestrator.py 使用 StateGraph 构建有状态工作流，编译时挂载 AsyncSqliteSaver 检查点（按 thread_id 持久化），支持 ainvoke 流式输出。

4.1 状态机节点
节点	职责
decomposer_node	意图识别（research/analysis）、复杂度分类（simple/complex）、领域检测（finance/contract/law/general）及子任务拆分。
knowledge_retriever_node	在编排前预检索内部知识库，并结合用户权限进行过滤。
router_node	设置 current_agent，并触发流式「路由」事件。
review_pipeline_node	Map-Reduce 审查管线（领域自适应），finance/contract/law 各有专用管线。
agentic_research_node	复杂问题走 ReAct 模式（create_react_agent + bind_tools），LLM 自主决定工具调用。
research_node	简单问题走快速通道（KB 检索 → 联网搜索 → LLM 合成），支持流式 Token。
analysis_node	调用 AnalysisAgent，产出统计报告和可视化 Spec（Vega-Lite）。
aggregator_node	汇总各 Agent 结果到 final_response，处理空结果或错误降级。
4.2 关键工程特性
流式可观测：每个阶段通过 get_stream_writer() 推送 phase 事件，前端可实时展示进度。

语义缓存优先：research_node 先查跨用户语义缓存，命中并校验权限后直接返回。

高相似度直出：知识库检索 Score ≥ 0.85 时，跳过 LLM 合成，直接返回原文（带 📄 来源标注）。

代码级护栏：审查输出强制包含结构标记（如量化指标对照表），缺失则自动补正；禁止使用“表面上看”、“似乎”、“可能”等模糊词。

确定性前置提取：审查任务先从附件中自动提取并比对量化指标，再注入 LLM 上下文。

优雅降级：审查管线或 ReAct 失败时，均回退到快速通道或纯 LLM，确保有输出。

5. 多专业化 Agent
5.1 BaseAgent (BDI 基类)
位于 src/agents/base_agent.py，提供 perceive/deliberate/plan 的默认实现和 BDI 三件套，并集成消息总线功能（send_to / request_from / broadcast）。

子类需实现抽象方法 act()。

5.2 ResearchAgent (研究 Agent)
位于 src/agents/research_agent.py。

快速通道：KB 检索（Qdrant+BM25） → 联网搜索（Baidu AI → Tavily 兜底） → LLM 合成（_synthesize / _synthesize_stream）。

自主路径：使用 LangGraph create_react_agent + bind_tools([kb_search_tool, web_search_tool, calculate_tool])，LLM 自决搜索策略。

来源标注：知识库内容标 📄，网络内容标 🌐，严禁混淆。

韧性：LLM 调用带重试（@retry），失败时回退到 _fallback_synthesize 或快速通道。

5.3 AnalysisAgent (分析 Agent)
位于 src/agents/analysis_agent.py。

执行流程：提取数据点 → 计算统计量 → 检测趋势 → 生成可视化 Spec → 提取洞察 → 汇编报告。

内容适应性护栏：仅当问题包含数据类关键词（如“收入”、“利润”、“趋势”、“KPI”）时才执行分析，否则返回空，避免污染合同/法律类回答。

产出：生成 Vega-Lite 兼容的可视化 Spec，并将结果回写 BeliefBase（category="analysis_report"）。

5.4 ReviewerAgent / 审查管线
审查任务由 review_pipeline_node 调度 src/review_pipeline.py 中的专用管线，与通用 ReAct 相区分，保证审查结论的结构化与可追溯性。

6. 全局状态与配置
AgentState (状态 Schema)
src/state.py 定义了 LangGraph 的共享状态（TypedDict，字段可选），关键字段如下：

字段	说明
question / raw_input / user_name	输入问题与当前用户（用于权限过滤）。
intent / complexity / domain	编排决策依据。
current_agent / task_description / sub_tasks	路由与任务描述。
long_term_context	跨会话长期记忆（由 memory.long_term 注入）。
retrieved_context	检索到的文档片段（含 doc_id/text/score）。
research_report / analysis_result / analysis_visualization	各 Agent 的输出。
review_extraction_context	确定性前置提取的量化指标比对结果。
final_response / error / trace_id / thread_id	聚合结果、追踪 ID 与检查点 ID。
7. RAG 混合检索（数据层）
完整的检索链路位于 src/rag/、src/tools.py 和 src/semantic_cache.py：

切块：使用标题层级感知切割（chunker.py），差异化解析 Markdown/DOCX/PDF。

双路索引：

BM25（关键词）：bm25_index.py

稠密向量：Qdrant 本地磁盘模式（dense_index.py）

权限预过滤：检索时通过 get_user_accessible_doc_paths(user_name) 仅查询可见文档（Payload 过滤），从结构上消除召回空洞。

融合与精排：

RRF（k=60） 融合两路结果。

CrossEncoder（reranker.py）进行精排，取 Top-K。

语义缓存：

使用 512 维 Embedding + FAISS 索引。

跨用户共享，但进行引用文档权限的二次过滤。

新鲜度保障：缓存绑定源文件的 mtime，文件变更即惰性失效。全局指纹（监控 indexes.db）变更时清空缓存。

8. API 参考（Django 端点）
8.1 核心端点
方法	路径	说明
POST	/api/workflow	主入口：提交用户问题，触发 LangGraph 工作流（流式 SSE）。
POST	/api/workflow/resume	基于 thread_id 恢复会话/确认后重跑。
POST	/api/rag/search	直接检索知识库，返回权限过滤后的结果。
POST	/api/upload	上传文档并触发索引（含 visibility 可见性设置）。
GET	/api/files	获取文件列表。
PATCH	/api/files/<filename>/visibility	修改文档可见性。
DELETE	/api/files/<filename>	删除文档（同步清理索引与缓存）。
GET	/api/health	健康检查（分层检查：LLM/RAG/磁盘等）。
GET	/api/auto-health	健康巡检，同时检查 ERROR 和 WARNING 级别状态。
GET	/api/metrics / /api/graph	运行指标 / 工作流图结构。
POST	/api/auth/login / /api/auth/logout	登录 / 登出。
GET	/api/conversations	获取会话列表。
GET	/api/logs / /api/system/logs	获取运行日志 / 系统日志。
GET/POST	/api/autoheal/*	自动修复引擎（状态/启动/扫描/学习/缓存）。
GET/DELETE	/api/cache/*	语义缓存管理（列表/清空/删除条目/按域删除）。
GET	/api/admin/users	用户与角色管理（列表/创建）。
8.2 请求示例
bash
# 触发工作流（流式）
curl -N -X POST http://localhost:8000/api/workflow \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"question": "请审查这份授信报告的风险点", "top_k": 10}'

# 健康检查
curl http://localhost:8000/api/health
8.3 健康检查分层
src/core/health.py 的 check_health() 并发检查：

Critical：LLM 可用性（任一配置的 Provider 可用即视为 Healthy）。

Non-critical：RAG 索引器可用性、磁盘剩余空间（<100MB 告警）、可选 API（如 Tavily/Baidu）配置。

聚合状态：

unhealthy：任一 Critical 检查失败。

degraded：任一 Non-critical 检查异常。

healthy：所有检查通过。

9. 项目结构
text
src/
├── config.py              # 配置管理（环境变量 + 默认值）
├── state.py               # LangGraph 共享状态 AgentState
├── middleware.py          # 请求鉴权 / Token 计量中间件
├── tools.py               # LangChain 工具（kb_search/web_search/calculate）
├── permissions.py         # 基于角色的文档可见性模型
├── semantic_cache.py      # 语义缓存（权限过滤 + 指纹失效）
├── resilience.py          # 重试 / 熔断装饰器
├── review_pipeline.py     # 领域自适应审查管线
├── review_extractor.py    # 确定性量化指标提取与比对
├── prompts.py             # Prompt 模板
├── core/                  # BDI 认知核心
│   ├── belief_base.py
│   ├── goal_manager.py
│   ├── plan_executor.py
│   ├── message_bus.py
│   ├── state_manager.py
│   ├── app.py             # MASApplication 生命周期编排
│   ├── health.py
│   ├── exceptions.py
│   └── logging_config.py
├── agents/                # 多专业化 Agent
│   ├── base_agent.py
│   ├── research_agent.py
│   ├── analysis_agent.py
│   ├── reviewer_agent.py
│   └── registry.py
├── workflows/
│   └── orchestrator.py    # LangGraph 工作流引擎
├── rag/                   # 检索层
│   ├── indexer.py
│   ├── dense_index.py
│   ├── bm25_index.py
│   ├── reranker.py
│   ├── embedder.py
│   ├── chunker.py
│   ├── parser.py
│   └── hybrid_index.py
└── memory/
    └── long_term.py       # 跨会话长期记忆

core/                      # Django 应用（服务层）
├── views.py
├── urls.py
├── models.py
└── ...
注意：根目录下的 core/（Django 服务层）与 src/core/（BDI 认知核心）是两个不同目录，请勿混淆。

10. 快速启动
10.1 环境要求
Python 3.11+（推荐 3.12）

可访问的 LLM（OpenAI / DeepSeek API Key，或本地 Ollama）

可选：Redis（未安装则自动降级为内存模式）

10.2 安装与配置
bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，至少填入 LLM_PROVIDER 与对应 API Key

# 3. 启动服务（ASGI）
NO_PROXY='*' uvicorn backend.asgi:application --host 0.0.0.0 --port 8000
# 或使用 Django 开发服务器
python manage.py runserver 0.0.0.0:8000

# 4. 健康检查
curl http://localhost:8000/api/health

# 5. 首次使用：上传文档并构建索引
python reindex.py
10.3 前端
bash
cd frontend
npm install
npm run dev      # Vite 开发服务器，默认 :5173
# 生产构建
npm run build    # 输出到 dist/
11. 部署与运维
容器化：仓库包含 Dockerfile.prod 与 docker-compose.yml，可一键编排后端及依赖。

进程模型：ASGI 单进程 + asyncio.to_thread 包裹同步检索，确保多用户并发时不阻塞事件循环。

索引重建：文档变更后执行 python reindex.py，会清理旧索引并重建 Qdrant + BM25。

自动修复：/api/autoheal/* 提供后台扫描、自学习修复方案（缓存于 indexes/autoheal_llm_cache.json，避免重复 LLM 调用）。

日志：内存环形缓冲 + 按日轮转文件日志（logs/），可通过 /api/logs 查看。

水平扩展：LangGraph 检查点持久化到 SQLite（indexes/checkpoints.db），多副本需将检查点与索引置于共享存储；StateManager 可切换至 Redis 实现跨进程状态共享。

12. 安全与权限
认证：/api/auth/login 签发 Token，受保护端点经中间件鉴权（request.user_name）。

文档权限模型：共 6 级可见性（admin / legal_lead / hr_lead / legal / hr / public），由 get_user_accessible_doc_paths() 返回三态（全部 / 空集 / 可见集合）。

不越权三重保障：

检索层 Payload 预过滤（仅查可见文档）。
缓存层 _entry_accessible_to 按引用文档二次校验。
匿名请求不继承“所有文档”权限（owner != "anonymous" 守卫）。
来源标注：知识库内容标 📄，联网结果标 🌐，防止模型混淆来源。

输入护栏：审查输出经结构化校验并禁用模糊词（如“可能”、“似乎”），确保结论明确（达标/不达标、合规/不合规）。
