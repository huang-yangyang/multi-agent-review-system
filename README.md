# Multi-Agent System

BDI 认知架构 + 多专业化 Agent + LangGraph 工作流引擎 + FastAPI 后端。

## 架构概览

```
用户请求 → FastAPI → LangGraph Orchestrator
                         ├── decomposer_node (任务分解)
                         ├── router_node (Agent路由)
                         ├── research_node (研究Agent)
                         ├── analysis_node (分析Agent)
                         ├── customer_service_node (客服Agent)
                         └── aggregator_node (结果聚合)
```

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境
cp .env.example .env
# 编辑 .env 填入 API Key

# 3. 启动服务
python main.py

# 4. 测试
curl http://localhost:8000/health
```

## API 端点

| Method | Path | 说明 |
|--------|------|------|
| POST | /chat | 通用对话入口 |
| POST | /analyze | 数据分析 |
| POST | /research | 研究查询 |
| POST | /customer | 客服对话 |
| GET | /health | 健康检查 |

## 项目结构

```
src/
├── config.py              # 配置管理
├── state.py               # 全局状态定义
├── core/
│   ├── belief_base.py     # Belief: 知识库 + 环境感知
│   ├── goal_manager.py    # Desire: 目标管理 + 优先级调度
│   ├── plan_executor.py   # Intention: 计划生成 + 执行
│   ├── message_bus.py     # 异步消息总线
│   └── state_manager.py   # 分布式状态管理
├── agents/
│   ├── base_agent.py      # BDI Agent 基类
│   ├── research_agent.py  # 研究Agent
│   ├── analysis_agent.py  # 分析Agent
│   └── customer_service_agent.py  # 客服Agent
├── workflows/
│   └── orchestrator.py    # LangGraph 工作流引擎
└── api/
    └── routes.py          # FastAPI 路由
```
