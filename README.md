1. 项目概述
  本系统是一个面向文档审查与知识问答的多智能体（Multi-Agent）系统，核心能力：
  BDI 认知架构：每个 Agent 内置 Belief（信念库）/ Desire（目标管理）/ Intention（计划执行），形成 感知 → 慎思 → 计划 → 行动 的认知闭环。
  LangGraph 工作流引擎：用有状态图（StateGraph）编排多 Agent 协作，支持条件路由、检查点（checkpoint）持久化、流式 token 输出。
  领域自适应：自动识别 finance（授信/风控）、contract（合同）、law（法律）、general 领域，审查类问题走确定性 Map-Reduce 管线。
  混合检索（RAG）：BM25 关键词 + 向量稠密检索（Qdrant 本地模式）+ RRF 融合 + CrossEncoder 精排 + 语义缓存。
  权限隔离：基于角色的文档可见性模型，检索层做 pre-filter，缓存层做二次校验。
  技术栈
        层                                   选型
        工作流编排                  LangGraph（StateGraph + AsyncSqliteSaver 检查点）
        Agent 框架                 自研 BDI 基类 + LangChain create_react_agent（复杂路径）
        LLM                       OpenAI / DeepSeek / 本地模型（可切换，带熔断与 fallback）
        向量检索                   Qdrant（本地磁盘模式）+ BM25（关键词）
        认知知识库                  ChromaDB（向量）+ SQLite（兜底）
        状态/通信                   Redis（可选，自动降级内存）+ 内存异步消息总线
        Web 服务                   Django + ASGI（uvicorn）
        前端                         Vue 3 + Vite
   
3. BDI 认知架构
  BDI（Belief-Desire-Intention）是 agent 认知建模的经典范式，本系统将其落地为三个可复用的核心模块，并由 BaseAgent 串成认知循环。
        BDI 要素                    对应模块                             职责
     Belief（信念）      src/core/belief_base.py          知识/感知存储与检索（ChromaDB 向量 + SQLite 兜底，带 TTL、分类）
     Desire（愿望）      src/core/goal_manager.py         目标生命周期管理（优先级队列、状态机、层级分解）
     Intention（意图）   src/core/plan_executor.py        目标→步骤分解、依赖图、分步执行与进度跟踪
   
2.1 认知循环  
  所有 Agent 继承 BaseAgent，其 run() 实现标准闭环：
  perceive（感知）  →  更新 BeliefBase（观察环境、记录用户输入）
  deliberate（慎思）→  基于 Belief 创建 Goal（GoalManager）
  plan（计划）      →  为激活的 Goal 生成执行 Plan（PlanExecutor）
  act（行动）       →  执行 Plan 步骤，产出结果（子类实现）
  flowchart TD
      A[用户输入] --> B[perceive: 写入 BeliefBase]
      B --> C[deliberate: 创建 Goal]
      C --> D[plan: 生成执行步骤]
      D --> E[act: 执行并产出]
      E --> F[结果返回 Orchestrator]
   
2.2 BeliefBase（信念库）
    双后端：ChromaDB（语义向量检索，优先）+ SQLite（关键词兜底，LIKE 匹配）。
    数据带 category 标签（如 perception / message / analysis_report），支持按类过滤。
    支持 belief_ttl（默认 86400s）自动过期，避免信念陈旧。
    对外 API：add_knowledge / query / get_context（拼接为 LLM 上下文）/ get_by_id / list_by_category / delete / clear_category / count。
    
  2.3 GoalManager（目标管理 / Desire）
    Goal 数据类：含 priority（HIGH=1/MEDIUM=2/LOW=3）、status（pending→active→completed/failed）、parent_goal_id + sub_goals（支持层级分解）。
    调度：get_next_goal() 按 HIGH→MEDIUM→LOW 取最高优先级待处理目标。
    容量限制：max_goals（默认 20），超限拒绝新建。
    父子完成联动：子目标全部完成后自动标记父目标完成。
    
  2.4 PlanExecutor（计划执行 / Intention）
    Plan / PlanStep 数据类；步骤支持 dependencies（依赖其他步骤）。
    generate_plan()：显式步骤或按 ;、\n 启发式拆分；默认线性依赖链，也可指定依赖图。
    execute_step() / get_ready_steps()：依赖满足的步骤可并行执行。
    get_progress()：返回 total/completed/failed/pending 与完成百分比。
    
  2.5 MessageBus（Agent 间通信）
    src/core/message_bus.py 提供完全异步的四种通信模式：
    P2P：send_p2p(sender, recipient, payload)
    PubSub：publish / subscribe / get_subscribers
    RequestResponse：request / send_response（基于 correlation_id 关联，带超时）
    Broadcast：broadcast（发送给所有注册 Agent）
    消息信封 Message 支持 HMAC 签名（verify_signature）保证完整性；默认内存交付，无需外部中间件。
  
  2.6 StateManager（分布式状态）
    src/core/state_manager.py：
    后端 Redis（可用时）+ 内存兜底（Redis 不可用时自动降级）。
    乐观并发：compare_and_set(key, expected_version, new_value) 基于版本号 CAS。
    支持 TTL、mget/mset 批量、get_by_prefix/delete_by_prefix 命名空间操作、JSON 序列化。

3. 系统总体架构
   分层职责
    1、接入层：Django 提供 REST 端点，承载鉴权、会话、文件上传、流式 SSE 响应。
    2、编排层：LangGraph Orchestrator 负责任务分解、知识预检索、路由与结果聚合。
    3、Agent 层：专业化 Agent 执行检索/分析/审查，内部走 BDI 认知闭环。
    4、认知核心层：BDI 三件套 + 消息总线 + 状态管理，供所有 Agent 复用。
    5、数据与检索层：混合 RAG 索引、文档库、语义缓存、权限模型。

4. LangGraph 工作流编排引擎
  src/workflows/orchestrator.py 使用 StateGraph 构建有状态工作流，编译时挂载 AsyncSqliteSaver 检查点（按 thread_id 跨会话持久化），支持 ainvoke 流式输出。

  4.1 状态机（实际节点
          节点                                                             职责
    decomposer_node                            意图识别（research/analysis）、复杂度分类（simple/complex）、领域检测（finance/contract/law/general）、子任务拆分；输出 intent/complexity/domain
    knowledge_retriever_node                   编排前预检索内部知识库（knowledge_search），并做用户权限过滤
    router_node                                      设置 current_agent，触发流式「路由」事件
    review_pipeline_node                       Map-Reduce 审查管线（领域自适应）：确定性提取指标/条款 → 单次 LLM 深度分析；finance/contract/law 各有专用管线
    agentic_research_node                      复杂问题走 ReAct（create_react_agent + bind_tools），LLM 自主决定工具调用
    research_node                              简单问题走快速通道（固定 3 阶段：KB 检索 → 联网搜索 → LLM 合成），支持流式 token
    analysis_node                                    调用 AnalysisAgent，产出统计报告 + 可视化 spec（Vega-Lite）
    aggregator_node                            汇总各 Agent 结果到 final_response，处理空结果/错误降级

  4.2 关键工程特性
    流式可观测：每个阶段通过 get_stream_writer() 推送 phase 事件（如 kb_search / web_search / token / review_map_start），前端可实时展示进度。
    语义缓存优先：research_node 先查语义缓存（跨用户共享 + 按引用文档权限过滤），命中直接返回。
    高相似度直出：知识库检索 score ≥ 0.85 时跳过 LLM 合成，直接返回原文（带 📄 来源标注）。
    代码级护栏：审查输出强制包含结构标记（量化指标对照表、抵押率对照、审批权限核查等），缺失则自动补正；禁止「表面上看/似乎/可能」等模糊词。
    确定性前置提取：审查任务先从附件报告与操作规程中自动提取并比对量化指标，注入 LLM 上下文。
    优雅降级：审查管线/ReAct 失败均回退到快速通道或纯 LLM，保证总有输出。

5. 多专业化 Agent
  5.1 BaseAgent（BDI 基类）
    src/agents/base_agent.py：抽象 act()，提供 perceive/deliberate/plan 默认实现，内置 BDI 三件套与消息总线集成（send_to / request_from / broadcast）。
  5.2 ResearchAgent（研究 Agent）
    src/agents/research_agent.py：
    快速通道（act）：KB 检索（Qdrant+BM25）→ 联网搜索（Baidu AI → Tavily 兜底）→ LLM 合成（_synthesize / _synthesize_stream）。
    自主路径（_agentic_search）：LangGraph create_react_agent + bind_tools([kb_search_tool, web_search_tool, calculate_tool])，LLM 自决搜索策略（ReAct）。
    来源标注规则：知识库内容标 📄，网络内容标 🌐，严禁混淆。
    韧性：LLM 调用带重试（@retry），失败回退到 _fallback_synthesize 或快速通道。
  5.3 AnalysisAgent（分析 Agent）
    src/agents/analysis_agent.py：
  6 阶段：_extract_data_points → _compute_statistics → _detect_trends → _generate_visualization_spec → _extract_insights → 汇编报告。
    内容适应性护栏：仅当问题含数据类关键词（收入/利润/用户/趋势/KPI/数据…）才执行统计；否则返回空结果，避免污染合同/法律类回答。
    产出 Vega-Lite 兼容的可视化 spec（前端可渲染图表）。
    结果回写 BeliefBase（category="analysis_report"）。
  5.4 ReviewerAgent / 审查管线
    审查任务（finance/contract/law）由 review_pipeline_node 调度 src/review_pipeline.py 的专用管线，区别于通用 ReAct，保证结构化、可追溯的审查结论。

6. 全局状态与配置
  6.1 AgentState（状态 schema）
    src/state.py 定义 LangGraph 共享状态（TypedDict, total=False，字段级合并）：
      字段                                                                          说明
    question / raw_input / user_name                                    输入问题与当前用户（用于权限过滤）
    intent / complexity / domain                                                编排决策依据
    current_agent / task_description / sub_tasks                                路由与任务描述
    long_term_context                                                  跨会话长期记忆（由 memory.long_term 注入）
    retrieved_context                                                  检索到的文档片段（含 doc_id/text/score）
    research_report / analysis_result / analysis_visualization                  各 Agent 输出
    review_extraction_context                                            确定性前置提取的量化指标比对结果
    final_response / error / trace_id / thread_id                            聚合结果、追踪与检查点

7. RAG 混合检索（数据层）
  检索链路（src/rag/、src/tools.py、src/semantic_cache.py）：
  切块：标题层级感知切割（chunker.py），Markdown/DOCX/PDF 差异化解析。
  双路索引：BM25（关键词，bm25_index.py）+ 稠密向量（Qdrant 本地磁盘模式，dense_index.py）。
  权限 pre-filter：检索时按 get_user_accessible_doc_paths(user_name) 仅查可见文档（payload 过滤），从结构上消除大体量下的 RRF=0 召回空洞。
  融合与精排：RRF（k=60）融合两路 → CrossEncoder（reranker.py）精排取 top-k。
  语义缓存：512 维 embedding + FAISS 索引（缓存自身），跨用户共享，按引用文档权限二次过滤；全局指纹（监控 indexes.db）变更即清空。
  新鲜度保障：缓存绑定源文件 mtime，文件被改则对应缓存惰性失效。

8. API 参考（Django 端点）
   
  8.1 核心端点
  
        方法                               路径                                                                   说明
        POST                          /api/workflow                                          主入口：提交用户问题，触发 LangGraph 工作流（流式 SSE）
        POST                       /api/workflow/resume                                              基于 thread_id 恢复会话/确认后重跑
        POST                          /api/rag/search                                          直接检索知识库（指定 query/top_k，返回带权限过滤的结果）
        POST                          /api/upload                                                上传文档并触发索引（含 visibility 可见性）
        GET                            /api/files                                                                文件列表
        PATCH              /api/files/<filename>/visibility                                                    修改文档可见性
        DELETE                  /api/files/<filename>                                                    删除文档（同步清理索引与缓存）
        GET                    /api/health · /health                                              健康检查（分层：LLM/RAG/磁盘/可选 API）
        GET                          /api/auto-health                                                 同时检查 ERROR+WARNING 的健康巡检
        GET                    /api/metrics · /api/graph                                                运行指标 / 工作流图结构
        POST            /api/auth/login · /api/auth/logout                                                      登录 / 登出
        GET                        /api/conversations                                                            会话列表
        GET                    /api/logs · /api/system/logs                                                    运行 / 系统日志
        GET/POST      /api/autoheal/status · /start · /scan · /learn · /cache                                   自动修复引擎
        GET/DELETE    /api/cache/list · /cache/clear · /cache/entry/<id> · /cache/domain/<d>                     语义缓存管理
        GET            /api/admin/users · /api/admin/users/create                                              用户与角色管理
        
    8.2 请求示例
      # 触发工作流（流式）
      curl -N -X POST http://localhost:8000/api/workflow \
        -H "Authorization: Bearer <TOKEN>" \
        -H "Content-Type: application/json" \
        -d '{"question": "请审查这份授信报告的风险点", "top_k": 10}'
      # 健康检查
      curl http://localhost:8000/api/health
      
    8.3 健康检查分层
      src/core/health.py 的 check_health() 并发检查：
      critical：LLM 可用性（任一 provider 配置即 healthy）
      non-critical：RAG 索引器可用性、磁盘剩余空间（<100MB 告警）、可选 API（Tavily/Baidu）配置
      聚合状态：unhealthy（任一 critical 失败）/ degraded（任一非 critical 异常）/ healthy。

      
9. 项目结构
    src/
    ├── config.py              # 配置管理（环境变量 + 默认值，单例 config）
    ├── state.py               # LangGraph 共享状态 AgentState（TypedDict）
    ├── middleware.py          # 请求鉴权 / token 计量中间件
    ├── tools.py               # LangChain 工具（kb_search/web_search/calculate）
    ├── permissions.py         # 基于角色的文档可见性模型
    ├── semantic_cache.py      # 语义缓存（跨用户 + 权限过滤 + 指纹失效）
    ├── resilience.py          # 重试 / 熔断装饰器
    ├── review_pipeline.py     # 领域自适应审查管线（finance/contract/law）
    ├── review_extractor.py    # 确定性量化指标提取与比对
    ├── prompts.py             # Prompt 模板（含 review_finance_system）
    ├── core/                  # BDI 认知核心
    │   ├── belief_base.py     # Belief：知识库（ChromaDB + SQLite）
    │   ├── goal_manager.py    # Desire：目标管理（优先级队列/状态机）
    │   ├── plan_executor.py   # Intention：计划生成与执行
    │   ├── message_bus.py     # 异步消息总线（P2P/PubSub/Req-Res/Broadcast + HMAC）
    │   ├── state_manager.py   # 分布式状态（Redis + 内存兜底 + CAS）
    │   ├── app.py             # MASApplication 生命周期编排（框架无关）
    │   ├── health.py          # 分层健康检查（框架无关）
    │   ├── exceptions.py      # 异常体系
    │   └── logging_config.py  # 结构化日志
    ├── agents/                # 多专业化 Agent
    │   ├── base_agent.py      # BDI Agent 基类（perceive→deliberate→plan→act）
    │   ├── research_agent.py  # 研究 Agent（快速通道 + ReAct 自主路径）
    │   ├── analysis_agent.py  # 分析 Agent（统计 + 可视化 spec）
    │   ├── reviewer_agent.py  # 审查 Agent
    │   └── registry.py        # Agent 注册表
    ├── workflows/
    │   └── orchestrator.py    # LangGraph 工作流引擎（8 节点状态机）
    ├── rag/                   # 检索层
    │   ├── indexer.py         # 双路索引编排（BM25 + Qdrant）
    │   ├── dense_index.py     # Qdrant 稠密向量（本地磁盘）
    │   ├── bm25_index.py      # BM25 关键词
    │   ├── reranker.py        # CrossEncoder 精排
    │   ├── embedder.py        # 嵌入模型
    │   ├── chunker.py         # 标题层级感知切块
    │   ├── parser.py          # 文档解析（MD/DOCX/PDF）
    │   └── hybrid_index.py    # 混合索引封装
    └── memory/
        └── long_term.py       # 跨会话长期记忆
    
    core/                      # Django 应用（服务层）
    ├── views.py               # REST 端点实现
    ├── urls.py                # 路由（由 FastAPI 迁移而来）
    ├── models.py              # 用户/会话/权限 ORM 模型
    └── ...
      说明：根目录另有 Django 工程的 core/（服务层），与 BDI 认知核心 src/core/ 是两个不同目录，请勿混淆。
10. 快速启动
    10.1 环境要求
      Python 3.11+（推荐 3.12）
      可访问的 LLM（OpenAI / DeepSeek API Key，或本地 Ollama）
      可选：Redis（不装则自动降级内存）
    
    10.2 安装与配置
      # 1. 安装依赖
      pip install -r requirements.txt
      
      # 2. 配置环境
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
      cd frontend
      npm install
      npm run dev      # Vite 开发服务器（默认 :5173）
      # 生产构建
      npm run build    # 输出到 dist/
    
11. 部署与运维
  容器化：仓库已包含 Dockerfile.prod 与 docker-compose.yml，可一键编排后端 + 依赖。
  进程模型：ASGI 单进程 + asyncio.to_thread 包裹同步检索，多用户并发时检索不阻塞事件循环。
  索引重建：文档变更后执行 python reindex.py（会清理旧索引并重建 Qdrant + BM25）。
  自动修复：/api/autoheal/* 提供后台扫描、自学习修复方案（indexes/autoheal_llm_cache.json 按 pattern 缓存，避免重复调用 LLM）。
  日志：内存环形缓冲 + 按日轮转文件日志（logs/），/api/logs 可查。
  水平扩展：LangGraph 检查点已持久化到 SQLite（indexes/checkpoints.db），多副本需将检查点与索引置于共享存储；StateManager 可切 Redis 实现跨进程状态共享。

12. 安全与权限
  认证：/api/auth/login 签发 token，受保护端点经中间件鉴权（request.user_name）。
  文档权限模型：6 级可见性（admin / legal_lead / hr_lead / legal / hr / public），由 get_user_accessible_doc_paths() 返回三态（全部 / 空集 / 可见集合）。
  不越权三重保障：
    检索层 payload pre-filter（仅查可见文档）；
    缓存层 _entry_accessible_to 按引用文档二次校验；
    匿名请求不继承「匿名所有文档」（owner != "anonymous" 守卫）。
  来源标注：知识库内容标 📄、联网结果标 🌐，防止模型混淆来源。
  输入护栏：审查输出结构化校验 + 模糊词禁用，保证结论明确（达标/不达标/合规/不合规）。
