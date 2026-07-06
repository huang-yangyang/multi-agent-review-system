<template>
  <div class="agent-flow-page">
    <!-- ── 页头 ── -->
    <div class="flow-header">
      <div>
        <h3>工作流架构 — 多领域 Map-Reduce + ReAct 双模式</h3>
        <p>Decomposer → Knowledge Retriever → Router → 审查/风险任务自动走 Map-Reduce（finance/contract/law 三领域自适应），其余走 ReAct/快速通道/分析 → Aggregator → END</p>
      </div>
      <div class="legend">
        <span class="legend-item"><i class="dot" style="background:#ff6b6b"></i>Map-Reduce 审查（穷举保证）</span>
        <span class="legend-item"><i class="dot" style="background:#22d3ee"></i>ReAct Tool Calling（自主推理）</span>
        <span class="legend-item"><i class="dot" style="background:#38bdf8"></i>快速通道（固定管线）</span>
        <span class="legend-item"><i class="dot" style="background:#8b5cf6"></i>拆解 / 检索 / 路由 / 聚合</span>
      </div>
    </div>

    <!-- ── ECharts 流程图 ── -->
    <div class="flow-diagram">
      <GraphVisualizer :active-node="selected" />
    </div>

    <!-- ── 路径 Tab 切换 ── -->
    <div class="path-tabs">
      <button v-for="tab in pathTabs" :key="tab.key"
        :class="['path-tab', { active: activePath === tab.key }]"
        :style="activePath === tab.key ? { '--tab-color': tab.color } : {}"
        @click="activePath = tab.key">
        <span class="tab-dot" :style="{ background: tab.color }"></span>
        {{ tab.label }}
      </button>
    </div>

    <!-- ── 当前路径的模块卡片 ── -->
    <div class="flow-details" :key="activePath">
      <div v-for="d in currentDetails" :key="d.role"
        class="detail-card" :class="{ highlight: d.highlight }"
        @mouseenter="selected = d.role"
        @mouseleave="selected = ''">
        <!-- 头部 -->
        <div class="detail-header">
          <div class="detail-icon" :style="{ background: d.gradient }">
            <el-icon :size="17"><component :is="d.icon" /></el-icon>
          </div>
          <div class="detail-titles">
            <span class="detail-role">{{ d.title }}</span>
            <el-tag :type="d.tagType" size="small" effect="dark" round>{{ d.status }}</el-tag>
          </div>
        </div>
        <!-- 描述 -->
        <p class="detail-desc">{{ d.desc }}</p>
        <!-- 技术栈 -->
        <div class="detail-tools" v-if="d.tools.length">
          <span class="tools-label">技术栈</span>
          <el-tag v-for="t in d.tools" :key="t" size="small" class="tool-tag" effect="plain">{{ t }}</el-tag>
        </div>
        <!-- 审查路径独有：D1-D6 指标徽章 -->
        <div v-if="d.metrics" class="detail-metrics">
          <span class="metrics-label">审查质量</span>
          <span v-for="m in d.metrics" :key="m" class="metric-badge" :class="m.includes('✅') ? 'pass' : 'fail'">{{ m }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import GraphVisualizer from '../components/GraphVisualizer.vue'
import {
  DataAnalysis, Cpu, Search, Share, ChatDotRound, TrendCharts, MagicStick, Checked, Connection,
} from '@element-plus/icons-vue'

const selected = ref('')
const activePath = ref('review')

// ── 节点点击 → 自动切 Tab ──
function onNodeClick(e) {
  const id = e.detail?.nodeId || ''
  selected.value = id
  const m = { review_pipeline:'review', agentic_research:'agentic', research_fast:'fast', analysis:'analysis' }
  if (m[id]) activePath.value = m[id]
}
onMounted(() => window.addEventListener('node-click', onNodeClick))
onBeforeUnmount(() => window.removeEventListener('node-click', onNodeClick))

// ── 4 条路径 Tab ──
const pathTabs = [
  { key:'review',  label:'审查任务 ★ Map-Reduce', color:'#ff6b6b' },
  { key:'agentic', label:'复杂研究 · ReAct',     color:'#22d3ee' },
  { key:'fast',    label:'简单研究 · 快速通道',   color:'#38bdf8' },
  { key:'analysis',label:'数据分析',              color:'#34d399' },
]

// ── 模块详情 ──
const allDetails = {
  // 前三个是所有路径共享的
  shared: [
    {
      role:'decomposer', title:'Decomposer — 意图识别 + 领域检测 + 复杂度判定',
      status:'入口', tagType:'',
      icon:DataAnalysis, gradient:'linear-gradient(135deg,#8b5cf6,#6366f1)',
      desc:'LLM 通过 Pydantic Structured Output 解析用户输入，输出三个维度的分类标签。intent 判定：检测 "analyze/分析/统计" → analysis，"research/研究/搜索" → research。complexity 判定：含 "比较/对比/分析/评估/建议/vs" 或多句复合提问 → complex。domain 判定：基于 26 个金融关键词（授信/风控/资产负债/抵押…）、20 个合同关键词、30 个法律关键词的加权匹配，命中 ≥2 个或唯一命中 ≥1 个时打标。附件内容会被剥离，仅对用户真正的问题文本做意图分类。',
      tools:['DeepSeek LLM','Pydantic 校验','26+20+30 关键词匹配','附件剥离']
    },
    {
      role:'knowledge_retriever', title:'Knowledge Retriever — 知识库预检索',
      status:'检索', tagType:'',
      icon:Search, gradient:'linear-gradient(135deg,#a78bfa,#8b5cf6)',
      desc:'在 Agent 执行前预先搜索本地知识库，将 top-5 文档片段作为 retrieved_context 注入后续节点。双路检索：FAISS 稠密向量索引（语义相似） + BM25 稀疏关键词索引（精确匹配）→ RRF 融合排序 → CrossEncoder 精排 → 按 domain 标签过滤 → 返回最相关的文档片段。目的：减少 Agent 后续的工具调用次数，降低延迟。',
      tools:['FAISS','BM25','RRF 融合','CrossEncoder','领域过滤']
    },
    {
      role:'router', title:'Router — 四维条件路由分发',
      status:'路由', tagType:'warning',
      icon:Share, gradient:'linear-gradient(135deg,#f59e0b,#d97706)',
      desc:'四维条件判定，将任务分发到最优路径。审查/风险任务自动走 Map-Reduce 管线，永不进入 ReAct 循环。触发条件：① finance + 审查关键词（审查/审核/检查）② contract + 审查/风险关键词（审查/风险/分析/评估/合规）③ law + 审查关键词。Map-Reduce 管线内部按领域自适应：金融用财务指标提取器，合同用条款-风险模式匹配器。',
      tools:['条件路由','多领域审查检测','4 路径分发']
    },
  ],

  // ★ Map-Reduce 审查
  review: [{
    role:'review_pipeline', title:'★ Review Pipeline — Map-Reduce 审查管线',
    status:'审查专用', tagType:'danger',
    icon:Checked, gradient:'linear-gradient(135deg,#ff6b6b,#ef4444)', highlight:true,
    desc:'审查/风险分析任务的核心引擎，按领域自适应选择 Map 提取器。金融领域（finance）：正则逐项提取 7 项财务指标 → 解析规程 4 张阈值表 → 逐项比对判定 → 输出结构化比对表。合同领域（contract）：提取全部合同条款 → 解析知识库全部风险模式 → 关键词预匹配 → 输出逐条预匹配清单。Reduce 阶段统一为单次 LLM 推理（无工具，非 ReAct），基于 Map 的结构化结果做最终语义判定和深度分析。LLM 不再需要自主决定"查什么"，只需要基于确定性数据做判断。',
    tools:['金融:正则指标提取','合同:条款+风险模式预匹配','LLM 单次推理(非ReAct)','领域自适应分发','3条降级路径'],
    metrics:['✅ 金融 D1-D6 全通过','✅ 合同条款全覆盖','✅ 风险模式穷举','✅ 来源分区正确','✅ 零模糊词汇'],
  }],

  // ReAct 推理
  agentic: [{
    role:'agentic_research', title:'Agentic Research — ReAct Tool Calling',
    status:'复杂推理', tagType:'success',
    icon:ChatDotRound, gradient:'linear-gradient(135deg,#22d3ee,#0ea5e9)',
    desc:'复杂多步推理任务的标准路径。使用 LangGraph create_react_agent + bind_tools 构建 Tool Calling Agent。LLM 在 ReAct 循环中自主决定：调用 kb_search_tool（本地知识库 FAISS+BM25）、web_search_tool（百度 AI 搜索 → Tavily 备用，各自独立熔断保护）、calculate_tool（安全表达式计算器）。最多 10 轮迭代，LLM 自主决定何时信息充足、开始生成最终回答。适用场景：开放域信息检索、多维度对比分析、"A vs B 哪个更好"类推荐。不适用场景：穷举比对类审查（→ 请用 Map-Reduce 路径）。',
    tools:['kb_search_tool','web_search_tool','calculate_tool','ReAct 循环 ≤10轮','bind_tools']
  }],

  // 快速通道
  fast: [{
    role:'research_fast', title:'Research — 快速通道（固定三阶段管线）',
    status:'轻量管道', tagType:'',
    icon:TrendCharts, gradient:'linear-gradient(135deg,#38bdf8,#0284c7)',
    desc:'简单问答/事实查询的固定管线，不做 ReAct 循环，无工具调用决策开销，延迟显著低于 ReAct 路径。Phase 1: 知识库搜索（Query Expansion 生成 2 条改写查询提升召回）→ Phase 2: 联网搜索（百度 → Tavily 备用，各独立熔断）→ Phase 3: LLM 合成（流式输出，按标题层级组织答案，来源分区标注 📄/🌐/🤖）。支持语义缓存：余弦相似度 ≥ 阈值时直接返回缓存答案，延迟接近 0。',
    tools:['Query Expansion','ES 搜索','联网搜索','LLM 合成','语义缓存','流式 SSE']
  }],

  // 数据分析
  analysis: [{
    role:'analysis', title:'Analysis — 数据分析管线',
    status:'分析节点', tagType:'success',
    icon:MagicStick, gradient:'linear-gradient(135deg,#34d399,#10b981)',
    desc:'用户问题含 "analyze/analysis/statistics/data/分析/统计" 关键词时触发。Analysis Agent 接收上游输出（如 research_report），进行结构化数据提取、统计计算和可视化图表生成。',
    tools:['统计分析','数据可视化','结构化提取']
  }],

  // 聚合（所有路径共享）
  aggregator: [{
    role:'aggregator', title:'Aggregator — 结果聚合 + 代码级护栏 + 缓存写入',
    status:'出口', tagType:'',
    icon:Connection, gradient:'linear-gradient(135deg,#818cf8,#6366f1)',
    desc:'① 合并各管线输出为 final_response。② 审查任务专用护栏：_validate_review_output() 代码级检查 — 8 项强制结构标记（量化指标对照表/抵押率/审批层级/📄分区等）是否完整、禁止模糊词汇（表面上看/似乎/可能/大概/貌似/或许）是否出现；缺失标记时自动补充降级框架。③ 写入语义缓存，后续相似问题可直接命中。',
    tools:['结果合并','_validate_review_output','降级补充','语义缓存写入']
  }],
}

const currentDetails = computed(() => {
  const cards = allDetails[activePath.value] || []
  return [...allDetails.shared.slice(0,3), ...cards, ...allDetails.aggregator]
})
</script>

<style scoped>
.agent-flow-page { display:flex; flex-direction:column; height:100%; }

/* ── 页头 ── */
.flow-header { display:flex; align-items:flex-end; justify-content:space-between; margin-bottom:10px; }
.flow-header h3 { font-size:19px; font-weight:700; }
.flow-header p { font-size:12.5px; color:var(--text-muted); margin-top:3px; max-width:680px; }
.legend { display:flex; gap:12px; flex-shrink:0; }
.legend-item { display:flex; align-items:center; gap:5px; font-size:11.5px; color:var(--text-muted); }
.legend-item .dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }

/* ── 图容器 ── */
.flow-diagram { flex:1; background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); margin-bottom:10px; min-height:380px; padding:2px; position:relative; overflow:hidden; }

/* ── 路径 Tab ── */
.path-tabs { display:flex; gap:5px; margin-bottom:10px; flex-shrink:0; flex-wrap:wrap; }
.path-tab { display:flex; align-items:center; gap:5px; padding:6px 13px; border:1px solid var(--border); border-radius:18px; background:var(--bg-card); color:var(--text-secondary); font-size:12px; cursor:pointer; transition:all .2s; white-space:nowrap; }
.path-tab:hover { border-color:var(--border-primary); color:var(--text-primary); }
.path-tab.active { border-color:var(--tab-color,#22d3ee); background:color-mix(in srgb,var(--tab-color,#22d3ee) 12%,transparent); color:var(--tab-color,#22d3ee); font-weight:600; box-shadow:0 0 12px color-mix(in srgb,var(--tab-color,#22d3ee) 20%,transparent); }
.tab-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }

/* ── 详情卡片 ── */
.flow-details { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:10px; flex-shrink:0; }
.detail-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:15px; transition:all .2s; cursor:default; }
.detail-card:hover { border-color:var(--border-primary); transform:translateY(-2px); box-shadow:var(--shadow-glow); }
.detail-card.highlight { border-color:#ff6b6b; box-shadow:0 0 20px rgba(255,107,107,.18); background:linear-gradient(135deg,rgba(255,107,107,.04),var(--bg-card)); }
.detail-header { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.detail-icon { width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:#fff; flex-shrink:0; }
.detail-titles { display:flex; flex-direction:column; gap:4px; }
.detail-role { font-weight:700; font-size:13px; }
.detail-desc { font-size:12px; color:var(--text-secondary); line-height:1.65; }
.detail-tools { margin-top:10px; display:flex; align-items:center; gap:5px; flex-wrap:wrap; }
.tools-label { font-size:10.5px; color:var(--text-muted); }
.tool-tag { font-size:10.5px; font-family:var(--font-mono); }
.detail-metrics { margin-top:8px; display:flex; align-items:center; gap:3px; flex-wrap:wrap; }
.metrics-label { font-size:10.5px; color:var(--text-muted); margin-right:3px; }
.metric-badge { font-size:10px; padding:1px 6px; border-radius:4px; font-weight:600; }
.metric-badge.pass { background:rgba(34,197,94,.15); color:#22c55e; }
.metric-badge.fail { background:rgba(239,68,68,.15); color:#ef4444; }

@media (max-width:1100px) { .flow-details { grid-template-columns:repeat(2,1fr); } }
@media (max-width:700px) { .flow-details { grid-template-columns:1fr; } .path-tabs { justify-content:center; } }
</style>
