<template>
  <div class="graph-container" ref="chartRef">
    <div class="graph-tooltip" v-if="tooltipNode" :style="tooltipStyle">
      <div class="tooltip-name">{{ tooltipNode.label }}</div>
      <div class="tooltip-desc">{{ tooltipNode.desc }}</div>
      <div class="tooltip-tags" v-if="tooltipNode.tags">
        <span v-for="t in tooltipNode.tags" :key="t" class="tooltip-tag">{{ t }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  activeNode: { type: String, default: '' }
})

const chartRef = ref(null)
const tooltipNode = ref(null)
const tooltipStyle = ref({})
let chart = null

// ═══════════════════════════════════════════════════════════════
// 架构节点 — 对应 src/workflows/orchestrator.py build_graph()
// ═══════════════════════════════════════════════════════════════

const NODE_DEFS = [
  // ── 入口 ──
  {
    id: 'start', label: 'START', desc: '用户通过聊天界面提交问题，可附带上传文档作为附件（自动注入到 question 文本的【附件：xxx】块中）',
    tags: ['用户输入', '附件上传'],
    x: 390, y: 45, color: '#22c55e', w: 110, h: 36,
    cat: 'terminal',
  },
  // ── 拆解层 ──
  {
    id: 'decomposer', label: 'Decomposer\n意图识别 · 领域检测 · 复杂度判定',
    desc: 'Pydantic Structured Output 约束 LLM 输出，解析用户问题 → 输出三要素：\n① intent: research | analysis\n② complexity: simple | complex\n③ domain: finance | contract | law | general\n\n领域检测基于 26 个金融关键词 + 20 个合同关键词 + 30 个法律关键词的加权匹配。附件内容会被剥离，仅对用户真正的问题文本做意图分类，避免附件中的关键词误触发领域判定。',
    tags: ['LLM (DeepSeek)', 'Pydantic 校验', '关键词加权匹配', '附件剥离'],
    x: 390, y: 148, color: '#8b5cf6', w: 200, h: 60,
    cat: 'decomposer',
  },
  // ── 检索层 ──
  {
    id: 'knowledge_retriever', label: 'Knowledge Retriever\n知识库预检索',
    desc: 'FAISS 稠密向量索引 + BM25 稀疏关键词索引 → RRF 融合排序 → CrossEncoder 精排 → 按 domain 标签过滤 → 返回 top-5 文档片段作为 retrieved_context。\n\n目的：在执行 Agent 之前预先加载相关知识，减少 Agent 的工具调用次数。',
    tags: ['FAISS 向量索引', 'BM25 关键词', 'RRF 融合', 'CrossEncoder 精排', '领域过滤'],
    x: 390, y: 275, color: '#a78bfa', w: 200, h: 60,
    cat: 'retriever',
  },
  // ── 路由层 ──
  {
    id: 'router', label: 'Router\n条件路由分发',
    desc: '四维判定逻辑：\n① domain == "finance" AND 问题含审查关键词 → ★ Map-Reduce 审查管线\n② intent == "research" AND complexity == "complex" → ReAct Tool Calling\n③ intent == "research" AND complexity == "simple" → 快速通道\n④ intent == "analysis" → 数据分析\n\n每种任务类型进入其最优执行路径，互不干扰。',
    tags: ['条件路由', '审查关键词匹配', '意图分发', '4 条路径'],
    x: 390, y: 402, color: '#f59e0b', w: 200, h: 60,
    cat: 'router',
  },
  // ── 4 条并行执行路径 ──
  {
    id: 'review_pipeline', label: '★ Review Pipeline\nMap-Reduce 审查',
    desc: '审查任务专用管线，永不进入 ReAct 循环。\n\nMap 阶段（纯代码 ~580行）：正则逐项提取 7 项财务指标 → 解析规程 4 张阈值表（评级/抵押率/审批层级/预警）→ 逐项比对判定达标/不达标 → 输出结构化 Markdown 比对表\n\nReduce 阶段（单次 LLM，无工具）：基于 Map 比对表 → 7 章节深度推理（总体评价→不达标分析→抵押率合规→审批权限→贷后预警→定性补充→总体建议）',
    tags: ['Map-Reduce', '确定性正则提取', '穷举保证', 'LLM 单次推理', '流式输出'],
    x: 48, y: 548, color: '#ff6b6b', w: 224, h: 72,
    cat: 'review', highlight: true,
  },
  {
    id: 'agentic_research', label: 'Agentic Research\nReAct Tool Calling',
    desc: '复杂多步推理任务的标准 ReAct 路径。LLM 自主决定：调用哪个工具 → 传什么参数 → 何时停止。最多 10 轮迭代。\n\n可用工具：kb_search_tool（本地知识库）、web_search_tool（百度 → Tavily 备用）、calculate_tool（安全计算器）。\n\n适用场景：开放域检索、多维度对比分析、"A和B哪个更好"类推荐。不适用场景：穷举比对类审查任务。',
    tags: ['ReAct 循环', 'Tool Calling', 'bind_tools', 'kb_search', 'web_search', 'calculate'],
    x: 290, y: 548, color: '#22d3ee', w: 200, h: 60,
    cat: 'agentic',
  },
  {
    id: 'research_fast', label: 'Research\n快速通道',
    desc: '简单问答/事实查询的固定三阶段管线，不做 ReAct 循环，无工具调用决策开销。\n\nPhase 1: 知识库搜索\nPhase 2: 联网搜索（百度 → Tavily 备用）\nPhase 3: LLM 合成（流式输出）\n\n支持语义缓存命中：相似问题（余弦相似度 ≥ 阈值）直接返回缓存答案，延迟接近 0。',
    tags: ['固定管线', 'ES 搜索', '联网搜索', 'LLM 合成', '语义缓存', '流式输出'],
    x: 530, y: 548, color: '#38bdf8', w: 200, h: 60,
    cat: 'fast',
  },
  {
    id: 'analysis', label: 'Analysis\n数据分析',
    desc: '当用户问题包含 "analyze"/"分析"/"统计" 等关键词时触发。Analysis Agent 执行统计计算和数据可视化任务，生成分析报告和图表。',
    tags: ['统计分析', '数据可视化', '结构化提取'],
    x: 770, y: 548, color: '#34d399', w: 160, h: 56,
    cat: 'analysis',
  },
  // ── 聚合层 ──
  {
    id: 'aggregator', label: 'Aggregator\n结果聚合 · 护栏校验 · 缓存写入',
    desc: '① 合并各管线输出为 final_response\n② 审查任务专用：_validate_review_output() 代码级护栏 — 检查 8 项强制结构标记 + 禁止模糊词汇检测 → 缺失时自动补充降级框架\n③ 写入语义缓存（后续相似问题直接命中）',
    tags: ['结果合并', '护栏校验', '降级补充', '语义缓存写入'],
    x: 390, y: 690, color: '#818cf8', w: 200, h: 60,
    cat: 'aggregator',
  },
  // ── 出口 ──
  {
    id: 'end', label: 'END', desc: '最终审查报告 / 研究结果 / 分析报告返回用户聊天界面，支持流式逐段渲染',
    tags: ['SSE 流式', 'Markdown 渲染'],
    x: 390, y: 795, color: '#ef4444', w: 130, h: 36,
    cat: 'terminal',
  },
]

const EDGE_DEFS = [
  // 主链
  { source: 'start', target: 'decomposer' },
  { source: 'decomposer', target: 'knowledge_retriever' },
  { source: 'knowledge_retriever', target: 'router' },
  // 路由 → 4 条路径（颜色与目标节点一致）
  { source: 'router', target: 'review_pipeline',   label: '审查/风险任务', color: '#ff6b6b', width: 2.4 },
  { source: 'router', target: 'agentic_research',  label: '复杂研究',            color: '#22d3ee', width: 1.8 },
  { source: 'router', target: 'research_fast',     label: '简单研究',            color: '#38bdf8', width: 1.6 },
  { source: 'router', target: 'analysis',          label: '分析意图',            color: '#34d399', width: 1.5 },
  // 4 条路径 → 聚合
  { source: 'review_pipeline',  target: 'aggregator' },
  { source: 'agentic_research', target: 'aggregator' },
  { source: 'research_fast',    target: 'aggregator' },
  { source: 'analysis',         target: 'aggregator' },
  // 聚合 → 出口
  { source: 'aggregator', target: 'end' },
]

// ── 辅助 ──
function shade(h, a = -30) { const m = h.replace('#','').match(/.{2}/g); return m ? '#'+m.map(x=>Math.max(0,Math.min(255,parseInt(x,16)+a)).toString(16).padStart(2,'0')).join('') : h }
function lit(h, a = 20) { return shade(h, a) }

function buildOption() {
  const a = props.activeNode
  return {
    backgroundColor: 'transparent', animation: true, animationDuration: 500, animationEasing: 'cubicOut',
    series: [{
      type: 'graph', layout: 'none', roam: false, draggable: false,
      symbol: 'roundRect', symbolKeepAspect: false, cursor: 'pointer',
      emphasis: { focus: 'self', blurScope: 'coordinateSystem', scale: 1.06 },
      data: NODE_DEFS.map(n => {
        const act = n.id === a
        return {
          id: n.id, name: n.label, x: n.x, y: n.y,
          symbolSize: act ? [n.w+16, n.h+8] : [n.w, n.h],
          itemStyle: {
            color: n.cat === 'terminal' ? n.color
              : new echarts.graphic.LinearGradient(0,0,1,1,[{offset:0,color:lit(n.color,15)},{offset:1,color:shade(n.color,-25)}]),
            shadowBlur: act ? 28 : (n.highlight ? 16 : 8),
            shadowColor: n.color, shadowOffsetY: act ? 3 : 1,
            borderColor: act ? '#fff' : (n.highlight ? lit(n.color,45) : 'rgba(255,255,255,0.10)'),
            borderWidth: act ? 2.5 : (n.highlight ? 2 : 1),
            opacity: act ? 1 : 0.93, borderRadius: 10,
          },
          label: { show:true, fontSize: n.cat==='review'?11:10, fontWeight: n.highlight?700:600, lineHeight:14.5, color:'#fff', formatter: n.label },
          _desc: n.desc, _tags: n.tags,
        }
      }),
      links: EDGE_DEFS.map(e => ({
        source: e.source, target: e.target, symbol: ['none','arrow'], symbolSize: [0,9],
        label: e.label ? { show:true, formatter:e.label, fontSize:9.5, fontWeight:600, color:'#e6eaf5', backgroundColor:'rgba(18,24,45,0.92)', padding:[2,8], borderRadius:5, borderColor: e.color||'rgba(148,163,220,0.25)', borderWidth:1 } : { show:false },
        lineStyle: { color: e.color||'rgba(148,163,220,0.32)', width: e.width||1.5, type:'solid', curveness: e.source==='router'?0.02:0.04 },
      })),
    }],
  }
}

// ── 交互 ──
function onOver(p) {
  if (p.dataType==='node') { const d=NODE_DEFS.find(n=>n.id===p.data.id); if(d){ tooltipNode.value={label:d.label.replace(/\n/g,' · '),desc:d.desc,tags:d.tags}; tooltipStyle.value={left:p.event.offsetX+16+'px',top:p.event.offsetY-10+'px'} } }
}
function onOut() { tooltipNode.value = null }
function onClick(p) {
  if (p.dataType==='node') window.dispatchEvent(new CustomEvent('node-click',{detail:{nodeId:p.data.id}}))
}

function init() {
  if (!chartRef.value) return
  chart = echarts.init(chartRef.value, 'dark')
  chart.setOption(buildOption())
  chart.on('mouseover','series',onOver); chart.on('mouseout','series',onOut); chart.on('click','series',onClick)
  window.addEventListener('resize', ()=>chart?.resize())
}
watch(()=>props.activeNode, ()=>{ if(chart) chart.setOption(buildOption(),true) })
onMounted(()=>nextTick(init))
onBeforeUnmount(()=>{ chart?.dispose(); chart=null })
</script>

<style scoped>
.graph-container { width:100%; height:100%; min-height:420px; position:relative; }
.graph-tooltip { position:absolute; z-index:20; max-width:340px; background:rgba(12,17,36,0.97); border:1px solid rgba(99,102,241,0.45); border-radius:10px; padding:12px 15px; pointer-events:none; backdrop-filter:blur(10px); box-shadow:0 8px 32px rgba(0,0,0,0.55); }
.tooltip-name { font-weight:700; font-size:13px; color:#f1f5f9; margin-bottom:6px; }
.tooltip-desc { font-size:11.5px; color:#94a3b8; line-height:1.6; white-space:pre-line; }
.tooltip-tags { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
.tooltip-tag { font-size:10px; padding:2px 7px; border-radius:4px; background:rgba(99,102,241,0.22); color:#a5b4fc; font-family:'JetBrains Mono','Fira Code',monospace; }
</style>
