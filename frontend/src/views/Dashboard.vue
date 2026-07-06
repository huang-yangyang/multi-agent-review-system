<template>
  <div class="dashboard">
    <div class="dashboard-grid">
      <!-- 左：对话列表 -->
      <aside class="panel panel-left">
        <div class="panel-header">
          <span class="panel-title">
            <el-icon :size="16" color="var(--secondary)"><ChatDotRound /></el-icon>
            对话历史
          </span>
          <button class="new-btn" @click="createConversation()">
            <el-icon :size="13"><Plus /></el-icon> 新对话
          </button>
        </div>
        <div class="conversation-list" v-if="conversations.length">
          <div
            v-for="conv in conversations"
            :key="conv.id"
            class="conv-item"
            :class="{ active: conv.id === activeId }"
            @click="switchConversation(conv.id)"
            @contextmenu.prevent="openContextMenu($event, conv)"
          >
            <div class="conv-indicator" v-if="conv.id === activeId"></div>
            <div class="conv-main">
              <div class="conv-title" :class="{ 'conv-title-empty': !conv.title }">{{ conv.title || '未命名对话' }}</div>
              <div class="conv-preview">{{ conv.preview || '暂无消息' }}</div>
            </div>
            <button
              class="conv-menu-btn"
              @click.stop="openContextMenu($event, conv)"
              title="更多操作"
            >
              <el-icon :size="14"><MoreFilled /></el-icon>
            </button>
          </div>
        </div>
        <div v-else class="conv-empty">
          <el-icon :size="32" color="var(--text-muted)"><ChatDotRound /></el-icon>
          <p>暂无对话</p>
          <span>点击"新对话"开始</span>
        </div>

        <!-- 右键 / 三点菜单 -->
        <Teleport to="body">
          <div
            v-if="contextMenu.visible"
            class="context-overlay"
            @click="closeContextMenu"
          >
            <div
              class="context-menu"
              :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }"
              @click.stop
            >
              <div class="context-menu-item" @click="doPin(contextMenu.conv)">
                <el-icon :size="14"><Top /></el-icon>
                <span>{{ contextMenu.conv.pinned ? '取消置顶' : '置顶' }}</span>
              </div>
              <div class="context-menu-item" @click="startRename(contextMenu.conv)">
                <el-icon :size="14"><Edit /></el-icon>
                <span>重新命名</span>
              </div>
              <div class="context-menu-divider"></div>
              <div class="context-menu-item context-menu-danger" @click="doDelete(contextMenu.conv)">
                <el-icon :size="14"><Delete /></el-icon>
                <span>删除</span>
              </div>
            </div>
          </div>
        </Teleport>

        <!-- 重命名弹窗 -->
        <Teleport to="body">
          <div v-if="renameDialog.visible" class="dialog-overlay" @click="cancelRename">
            <div class="dialog-box" @click.stop>
              <h4 class="dialog-title">重新命名</h4>
              <el-input
                v-model="renameDialog.title"
                placeholder="输入对话名称"
                @keyup.enter="confirmRename"
                ref="renameInput"
              />
              <div class="dialog-actions">
                <button class="dialog-btn dialog-btn-cancel" @click="cancelRename">取消</button>
                <button class="dialog-btn dialog-btn-confirm" @click="confirmRename">确定</button>
              </div>
            </div>
          </div>
        </Teleport>

      </aside>

      <!-- 中：对话区 -->
      <section class="panel panel-center">
        <ChatInterface
          :messages="chatMessages"
          :loading="running && !interruptState"
          @send="handleSend"
          @cancel="handleCancel"
        />
      </section>

      <!-- 右：执行日志时间轴 -->
      <aside class="panel panel-right" v-if="chatMessages.length || logs.length">
        <div class="panel-header">
          <span class="panel-title">
            <el-icon :size="16" color="var(--secondary)"><Monitor /></el-icon>
            执行轨迹
          </span>
          <el-tag v-if="running" type="warning" size="small" effect="dark" round>运行中</el-tag>
          <el-tag v-else-if="logs.length" type="success" size="small" effect="dark" round>已完成</el-tag>
        </div>

        <div class="timeline" v-if="logs.length">
          <div v-for="(log, i) in logs" :key="i" class="timeline-item" :class="log.level">
            <div class="timeline-dot" :class="log.level">
              <el-icon :size="10">
                <component :is="logIcon(log)" />
              </el-icon>
            </div>
            <div class="timeline-body">
              <div
                class="timeline-agent"
                :class="{ 'has-substeps': log.substeps && log.substeps.length }"
                @click="log.substeps && log.substeps.length ? toggleExpand(log) : null"
              >
                <el-icon
                  v-if="log.substeps && log.substeps.length"
                  :size="10"
                  class="expand-arrow"
                  :class="{ expanded: log.expanded }"
                ><ArrowRight /></el-icon>
                {{ log.agent || logText(log) }}
                <span v-if="log.substeps && log.substeps.length" class="substep-count">{{ log.substeps.length }} 步</span>
              </div>
              <div class="timeline-detail" v-if="log.detail">{{ log.detail }}</div>
              <div class="timeline-sub" v-if="log.sub && log.sub.length">
                <span class="sub-chip" v-for="(q, qi) in log.sub" :key="qi">{{ qi + 1 }}. {{ q }}</span>
              </div>
              <!-- 子步骤展开 -->
              <div class="timeline-substeps" v-if="log.expanded && log.substeps.length">
                <div
                  v-for="(ss, si) in log.substeps"
                  :key="si"
                  class="substep-row"
                  :class="ss.status"
                >
                  <span class="substep-dot" :class="ss.status"></span>
                  <span class="substep-label">{{ ss.label }}</span>
                  <span class="substep-detail" v-if="ss.detail">{{ ss.detail }}</span>
                </div>
              </div>
              <div class="timeline-time">{{ log.time }}</div>
            </div>
          </div>
        </div>
        <div v-else class="log-empty">
          <el-icon :size="28" color="var(--text-muted)"><Monitor /></el-icon>
          <p>执行轨迹将在此显示</p>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import {
  ChatDotRound, Monitor, Plus, Delete,
  Connection, Cpu, DataAnalysis, Files, Loading, CircleCheckFilled, WarningFilled, ArrowRight, Checked,
  MoreFilled, Top, Edit
} from '@element-plus/icons-vue'
import ChatInterface from '../components/ChatInterface.vue'

const running = ref(false)
const abortController = ref(null)
const logs = ref([])
const conversations = reactive([])
const activeId = ref(null)
const chatMessages = ref([])
const renameInput = ref(null)

// Context menu state
const contextMenu = reactive({ visible: false, x: 0, y: 0, conv: null })

// Rename dialog state
const renameDialog = reactive({ visible: false, title: '', convId: null })

// HITL interrupt state
const interruptState = ref(null)          // { stage, question, prompt, ... }
const interruptThreadId = ref(null)       // thread_id for resume
const trackedAgent = ref('')              // 追踪当前执行路径：Map-Reduce 审查 / ReAct Agent / Research Agent

// ── helpers ───────────────────────────────────────
function authHeaders() {
  const token = localStorage.getItem('token')
  return token
    ? { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' }
}

function now() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false })
}

function logIcon(log) {
  if (log.level === 'error') return WarningFilled
  if (log.level === 'success') return CircleCheckFilled
  if (log.level === 'running') return Loading
  return Connection
}

function logText(log) {
  return log.text || ''
}

function toggleExpand(log) {
  log.expanded = !log.expanded
}

// 节点中文映射
const NODE_META = {
  decomposer_node: { name: '问题拆解', icon: DataAnalysis },
  router_node: { name: '模型路由', icon: Cpu },
  knowledge_retriever_node: { name: '知识检索', icon: Files },
  review_pipeline_node: { name: 'Map-Reduce 审查', icon: Checked },
  research_node: { name: 'ReAct 推理', icon: Cpu },
  agentic_research_node: { name: 'ReAct 推理', icon: Cpu },
  analysis_node: { name: '分析 Agent', icon: DataAnalysis },
  aggregator_node: { name: '结果聚合', icon: DataAnalysis },
}


// ── 加载对话列表 ──
async function loadConversations() {
  try {
    const r = await fetch('/api/conversations', { headers: authHeaders() })
    const data = await r.json()
    conversations.splice(0, conversations.length, ...(data.conversations || []))
    if (conversations.length && !activeId.value) {
      activeId.value = conversations[0].id
      await loadMessages(conversations[0].id)
    }
  } catch (e) {
    console.error('加载对话失败:', e)
  }
}

async function loadMessages(convId) {
  try {
    const r = await fetch(`/api/conversations/${convId}/messages`, { headers: authHeaders() })
    const data = await r.json()
    chatMessages.value = data.messages || []
  } catch (e) {
    console.error('加载消息失败:', e)
  }
}

// ── 对话管理 ──
async function createConversation() {
  const id = 'conv_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8)
  await fetch('/api/conversations', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ id, title: '' }),
  })
  conversations.unshift({ id, title: '', preview: '' })
  activeId.value = id
  chatMessages.value = []
  logs.value = []
}

async function switchConversation(id) {
  activeId.value = id
  await loadMessages(id)
  logs.value = []
}

async function deleteConversation(id) {
  await fetch(`/api/conversations/${id}/delete`, { method: 'DELETE', headers: authHeaders() })
  const idx = conversations.findIndex(c => c.id === id)
  if (idx !== -1) conversations.splice(idx, 1)
  if (activeId.value === id) {
    const next = conversations[Math.min(idx, conversations.length - 1)]
    if (next) {
      activeId.value = next.id
      await loadMessages(next.id)
    } else {
      activeId.value = null
      chatMessages.value = []
    }
  }
}

// ── 右键菜单 ──
function openContextMenu(event, conv) {
  const menuW = 160, menuH = 120
  let x = event.clientX
  let y = event.clientY
  if (x + menuW > window.innerWidth) x = window.innerWidth - menuW - 8
  if (y + menuH > window.innerHeight) y = window.innerHeight - menuH - 8
  contextMenu.x = x
  contextMenu.y = y
  contextMenu.conv = conv
  contextMenu.visible = true
}

function closeContextMenu() {
  contextMenu.visible = false
  contextMenu.conv = null
}

async function doPin(conv) {
  closeContextMenu()
  const newPinned = conv.pinned ? 0 : 1
  try {
    await fetch(`/api/conversations/${conv.id}`, {
      method: 'PATCH',
      headers: authHeaders(),
      body: JSON.stringify({ pinned: newPinned }),
    })
    conv.pinned = newPinned
    const idx = conversations.indexOf(conv)
    conversations.splice(idx, 1)
    if (newPinned) {
      const firstNonPinned = conversations.findIndex(c => !c.pinned)
      if (firstNonPinned === -1) conversations.push(conv)
      else conversations.splice(firstNonPinned, 0, conv)
    } else {
      const lastPinned = conversations.reduce((max, c, i) => c.pinned ? i : max, -1)
      conversations.splice(lastPinned + 1, 0, conv)
    }
  } catch (e) {
    console.error('置顶失败:', e)
  }
}

function startRename(conv) {
  closeContextMenu()
  renameDialog.convId = conv.id
  renameDialog.title = conv.title || ''
  renameDialog.visible = true
}

async function confirmRename() {
  const newTitle = renameDialog.title.trim()
  if (!newTitle || !renameDialog.convId) {
    cancelRename()
    return
  }
  try {
    await fetch(`/api/conversations/${renameDialog.convId}`, {
      method: 'PATCH',
      headers: authHeaders(),
      body: JSON.stringify({ title: newTitle }),
    })
    const conv = conversations.find(c => c.id === renameDialog.convId)
    if (conv) conv.title = newTitle
  } catch (e) {
    console.error('重命名失败:', e)
  }
  renameDialog.visible = false
  renameDialog.convId = null
  renameDialog.title = ''
}

function cancelRename() {
  renameDialog.visible = false
  renameDialog.convId = null
  renameDialog.title = ''
}

async function doDelete(conv) {
  closeContextMenu()
  await deleteConversation(conv.id)
}

// ── 保存消息 ──
async function saveMessage(role, content, agent = '', msgTime = '') {
  const convId = activeId.value
  if (!convId) return

  // 立即更新本地状态（避免被 Django 单线程阻塞延迟显示）
  const conv = conversations.find(c => c.id === convId)
  if (conv && role === 'user') {
    conv.title = content.slice(0, 30) + (content.length > 30 ? '...' : '')
  }
  if (conv && role === 'assistant' && content) {
    conv.preview = content.slice(0, 60) + (content.length > 60 ? '...' : '')
  }

  await fetch(`/api/conversations/${convId}/messages`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ role, content, agent, time: msgTime || now() }),
  })
}

// ── 取消 ──
function handleCancel() {
  if (abortController.value) {
    abortController.value.abort()
    abortController.value = null
  }
  interruptState.value = null
  interruptThreadId.value = null
  running.value = false
  const msg = '⏹️ 已取消本次请求'
  chatMessages.value.push({ role: 'assistant', content: msg, agent: 'System', time: now() })
  saveMessage('assistant', msg, 'System', now())
  logs.value.push({ level: 'warn', time: now(), agent: '系统', detail: '用户取消了请求' })
}

// ── 发送（支持 HITL interrupt 路由）──
async function handleSend(topic, attachments = []) {
  // If waiting for HITL interrupt response, route to resume
  if (interruptState.value && interruptThreadId.value) {
    return handleInterruptResponse(topic)
  }

  if (!activeId.value) await createConversation()

  running.value = true
  logs.value = []
  trackedAgent.value = ''  // 重置，等待 SSE 事件更新

  chatMessages.value.push({ role: 'user', content: topic, time: now() })
  saveMessage('user', topic, '', now())
  logs.value.push({ level: 'info', time: now(), agent: '用户提交', detail: topic })

  abortController.value = new AbortController()

  const body = { topic, stream: true, thread_id: activeId.value }
  if (attachments.length) {
    body.attachments = attachments
  }

  try {
    const r = await fetch('/api/workflow', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify(body),
      signal: abortController.value.signal,
    })
    if (!r.ok) {
      const err = (await r.json().catch(() => ({}))).detail || (await r.json().catch(() => ({}))).error || `请求失败 (${r.status})`
      throw new Error(err)
    }

    const reader = r.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const evt = JSON.parse(line.slice(6))
        handleSseEvent(evt)
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return
    const text = `执行失败: ${err.message}`
    chatMessages.value.push({ role: 'assistant', content: text, agent: 'System', time: now() })
    saveMessage('assistant', text, 'System', now())
    logs.value.push({ level: 'error', time: now(), agent: '异常', detail: err.message })
  }
  // Keep running=true if paused at HITL interrupt
  if (!interruptState.value) {
    running.value = false
  }
}

function handleSseEvent(evt) {
  if (evt.type === 'agent_event') {
    const meta = NODE_META[evt.agent] || { name: evt.agent }
    const out = evt.output || {}
    const entry = {
      level: out.status === 'done' ? 'success' : (out.status === 'running' ? 'running' : 'info'),
      time: now(),
      agent: meta.name,
      expanded: false,
      substeps: [],
    }
    if (out.preview) entry.detail = out.preview
    if (out.sub_questions) {
      entry.detail = `拆解为 ${out.sub_questions.length} 个子问题`
      entry.sub = out.sub_questions
    }
    if (out.iteration !== undefined) {
      entry.detail = (entry.detail ? entry.detail + ' · ' : '') + `第 ${out.iteration} 轮迭代`
    }
    logs.value.push(entry)
  } else if (evt.type === 'substep_event') {
    // 将子步骤附加到最近一个同 node 的日志条目
    const parentNode = NODE_META[evt.node] ? NODE_META[evt.node].name : evt.node
    for (let i = logs.value.length - 1; i >= 0; i--) {
      if (logs.value[i].agent === parentNode) {
        logs.value[i].substeps.push({
          step: evt.step,
          label: evt.label,
          detail: evt.detail,
          status: evt.status,
        })
        break
      }
    }
  } else if (evt.type === 'phase_event') {
    // 来自 research_node / agentic_research_node 的实时状态
    const phase = evt.phase
    const message = evt.message || ''
    const count = evt.count
    const content = evt.content
    const stage = evt.stage

    // 如果是 token 流式内容，追加到当前正在生成的回答
    if (phase === 'token' && content) {
      const lastMsg = chatMessages.value[chatMessages.value.length - 1]
      if (lastMsg && lastMsg.role === 'assistant') {
        lastMsg.content += content
        // 更新存储
        saveMessage('assistant', lastMsg.content, lastMsg.agent, lastMsg.time)
      } else {
        // 首次 token，使用追踪到的实际路径标签
        const agentName = trackedAgent.value || 'ReAct Agent'
        const newMsg = {
          role: 'assistant',
          content: content,
          agent: agentName,
          time: now(),
        }
        chatMessages.value.push(newMsg)
        saveMessage('assistant', content, agentName, now())
      }
      return
    }

    // ── 追踪实际执行路径（Map-Reduce vs ReAct vs 快速通道 vs 分析） ──
    if (phase === 'review_map_start' || phase === 'review_mode' || phase === 'review_extraction') {
      trackedAgent.value = 'Map-Reduce 审查'
    } else if (phase === 'agentic_start') {
      trackedAgent.value = 'ReAct Agent'
    } else if (phase === 'kb_search' || phase === 'synthesize') {
      if (!trackedAgent.value || trackedAgent.value === 'ReAct Agent') {
        trackedAgent.value = 'Research Agent'
      }
    } else if (phase === 'analysis_start') {
      trackedAgent.value = 'Analysis Agent'
    }

    // 其他状态日志
    const levelMap = {
      cache_hit: 'success',
      decomposer_done: 'info',
      kb_presearch_done: 'info',
      kb_search: 'running',
      kb_search_done: 'info',
      web_search: 'running',
      web_search_done: 'info',
      routing_done: 'info',
      synthesize: 'running',
      synthesize_done: 'success',
      agentic_start: 'running',
      agentic_done: 'success',
      analysis_start: 'running',
      analysis_done: 'success',
      review_map_start: 'running',
      review_map_done: 'success',
      review_reduce: 'running',
      review_done: 'success',
      review_mode: 'info',
      review_extraction: 'info',
      pipeline_stage: 'running',
      pipeline_stage_done: 'success',
      aggregating_done: 'info',
      error: 'error',
    }
    // Map phase to display agent label
    const agentLabelMap = {
      decomposer_done: '问题拆解',
      kb_presearch_done: '知识检索',
      routing_done: '模型路由',
      kb_search: 'Research Agent',
      kb_search_done: 'Research Agent',
      web_search: 'Research Agent',
      web_search_done: 'Research Agent',
      synthesize: 'Research Agent',
      synthesize_done: 'Research Agent',
      agentic_start: 'ReAct Agent',
      agentic_done: 'ReAct Agent',
      analysis_start: 'Analysis Agent',
      analysis_done: 'Analysis Agent',
      review_map_start: 'Map-Reduce 审查',
      review_map_done: 'Map-Reduce 审查',
      review_reduce: 'Map-Reduce 审查',
      review_done: 'Map-Reduce 审查',
      review_mode: 'Map-Reduce 审查',
      review_extraction: 'Map-Reduce 审查',
      cache_hit: '语义缓存',
      aggregating_done: '结果聚合',
    }
    const level = levelMap[phase] || 'info'
    const agentLabel = phase === 'pipeline_stage'
      ? `协作管线 · ${stage}`
      : (agentLabelMap[phase] || '状态更新')
    logs.value.push({
      level,
      time: now(),
      agent: agentLabel,
      detail: message,
      count,
    })
  } else if (evt.type === 'completed') {
    const text = evt.final_output || '（未获取到最终输出）'
    // 如果已有流式消息，更新其内容；否则新建
    const lastMsg = chatMessages.value[chatMessages.value.length - 1]
    if (lastMsg && lastMsg.role === 'assistant') {
      lastMsg.content = text
      saveMessage('assistant', text, lastMsg.agent, lastMsg.time)
    } else {
      chatMessages.value.push({
        role: 'assistant',
        content: text,
        agent: 'ReAct Agent',
        time: now(),
        iteration: evt.iteration,
        confidence: evt.confidence
      })
      saveMessage('assistant', text, 'ReAct Agent', now())
    }
    logs.value.push({
      level: 'success',
      time: now(),
      agent: '完成',
      detail: `答案已生成${evt.iteration !== undefined ? `（${evt.iteration} 轮迭代）` : ''}`
    })
    running.value = false
    interruptState.value = null
    interruptThreadId.value = null
  } else if (evt.type === 'error') {
    interruptState.value = null
    interruptThreadId.value = null
    logs.value.push({ level: 'error', time: now(), agent: '错误', detail: evt.error || '工作流异常' })
  } else if (evt.type === 'interrupt') {
    // HITL interrupt — show prompt and wait for user response
    const interrupts = evt.interrupts || []
    if (interrupts.length > 0) {
      const iv = interrupts[0].value || interrupts[0]
      interruptState.value = iv
      interruptThreadId.value = evt.thread_id
      // Show interrupt prompt as a system message in chat
      chatMessages.value.push({
        role: 'system',
        content: iv.prompt || '请确认是否继续',
        agent: iv.stage === 'confirm_topics' ? '子问题确认' : '答案审批',
        time: now(),
        _interrupt: true,
      })
      logs.value.push({
        level: 'running',
        time: now(),
        agent: iv.stage === 'confirm_topics' ? '子问题确认' : '答案审批',
        detail: `等待用户确认（意图: ${iv.detected_intent || '—'}, 知识库匹配: ${iv.kb_matches || 0} 条）`,
      })
    }
  }
}

// ── HITL Interrupt 响应 ──
async function handleInterruptResponse(response) {
  const threadId = interruptThreadId.value
  interruptState.value = null
  interruptThreadId.value = null

  // Show user's response in chat
  chatMessages.value.push({ role: 'user', content: response, time: now() })
  logs.value.push({ level: 'info', time: now(), agent: '用户确认', detail: response })

  abortController.value = new AbortController()

  try {
    const r = await fetch('/api/workflow/resume', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ thread_id: threadId, resume_value: response }),
      signal: abortController.value.signal,
    })
    if (!r.ok) {
      const err = (await r.json().catch(() => ({}))).error || `请求失败 (${r.status})`
      throw new Error(err)
    }

    const reader = r.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const evt = JSON.parse(line.slice(6))
        handleSseEvent(evt)
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return
    const text = `执行失败: ${err.message}`
    chatMessages.value.push({ role: 'assistant', content: text, agent: 'System', time: now() })
    saveMessage('assistant', text, 'System', now())
    logs.value.push({ level: 'error', time: now(), agent: '异常', detail: err.message })
  }
  // Keep running=true if paused at HITL interrupt
  if (!interruptState.value) {
    running.value = false
  }
}

onMounted(() => {
  loadConversations()
})
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  height: 100%;
  gap: 14px;
}

/* ===== 三栏布局 ===== */
.dashboard-grid {
  display: grid;
  grid-template-columns: 280px 1fr 300px;
  gap: 14px;
  flex: 1;
  min-height: 0;
}

.panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.panel-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.panel-title {
  display: flex; align-items: center; gap: 8px;
  font-weight: 600; font-size: 13px;
}
.panel-center { padding: 0 14px 14px; }
.panel-right { overflow-y: auto; }

.new-btn {
  display: flex; align-items: center; gap: 4px;
  padding: 5px 12px;
  background: var(--gradient-brand);
  border: none; border-radius: var(--radius-sm);
  color: #fff; font-size: 12px; font-weight: 600;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.new-btn:hover { filter: brightness(1.12); box-shadow: var(--shadow-glow); }

/* 对话列表 */
.conversation-list { flex: 1; overflow-y: auto; padding: 8px; }
.conv-item {
  position: relative;
  display: flex; align-items: center; gap: 10px;
  padding: 11px 12px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  margin-bottom: 4px;
  border: 1px solid transparent;
  transition: all var(--transition-fast);
}
.conv-item:hover { background: var(--bg-hover); }
.conv-item.active {
  background: var(--primary-dim);
  border-color: var(--border-primary);
}
.conv-indicator {
  position: absolute; left: 0; top: 50%; transform: translateY(-50%);
  width: 3px; height: 24px;
  background: var(--gradient-brand);
  border-radius: 0 3px 3px 0;
}
.conv-main { flex: 1; min-width: 0; }
.conv-title {
  font-size: 13px; font-weight: 600; color: var(--text-primary);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.conv-title-empty {
  font-style: italic;
  color: var(--text-muted);
  font-weight: 400;
}
.conv-preview {
  font-size: 11px; color: var(--text-muted); margin-top: 2px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.conv-menu-btn {
  width: 28px; height: 28px; border-radius: 6px;
  border: none; background: transparent;
  color: var(--text-muted); cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  opacity: 0; transition: all var(--transition-fast);
  flex-shrink: 0;
}
.conv-item:hover .conv-menu-btn { opacity: 1; }
.conv-menu-btn:hover { background: var(--bg-input); color: var(--text-primary); }

/* ── 右键/三点菜单 ── */
.context-overlay {
  position: fixed; inset: 0; z-index: 9999;
  background: transparent;
}
.context-menu {
  position: fixed; z-index: 10000;
  min-width: 150px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
  padding: 6px;
  animation: fadeIn 0.12s var(--ease-out);
}
@keyframes fadeIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
.context-menu-item {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
  color: var(--text-primary);
  cursor: pointer;
  transition: all var(--transition-fast);
}
.context-menu-item:hover { background: var(--bg-hover); }
.context-menu-danger { color: var(--danger); }
.context-menu-danger:hover { background: rgba(239,68,68,0.1); }
.context-menu-divider {
  height: 1px;
  background: var(--border);
  margin: 4px 8px;
}

/* ── 重命名弹窗 ── */
.dialog-overlay {
  position: fixed; inset: 0; z-index: 10001;
  background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center;
  animation: fadeIn 0.15s var(--ease-out);
}
.dialog-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  width: 360px;
  max-width: 90vw;
  box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.dialog-title {
  font-size: 15px; font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 16px;
}
.dialog-actions {
  display: flex; gap: 10px; justify-content: flex-end;
  margin-top: 18px;
}
.dialog-btn {
  padding: 7px 18px;
  border: none; border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 600;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.dialog-btn-cancel {
  background: var(--bg-input);
  color: var(--text-secondary);
}
.dialog-btn-cancel:hover { background: var(--bg-hover); }
.dialog-btn-confirm {
  background: var(--gradient-brand);
  color: #fff;
}
.dialog-btn-confirm:hover { filter: brightness(1.12); }

.conv-empty {
  padding: 60px 16px; text-align: center;
  display: flex; flex-direction: column; align-items: center; gap: 8px;
}
.conv-empty p { font-size: 13px; color: var(--text-secondary); font-weight: 600; }
.conv-empty span { font-size: 11px; color: var(--text-muted); }

/* ===== 时间轴 ===== */
.timeline {
  padding: 16px 14px;
  position: relative;
}
.timeline::before {
  content: ''; position: absolute;
  left: 24px; top: 20px; bottom: 20px;
  width: 1px; background: var(--border-strong);
}
.timeline-item {
  display: flex; gap: 12px; margin-bottom: 18px;
  position: relative;
  animation: slideIn 0.3s var(--ease-out);
}
@keyframes slideIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.timeline-dot {
  width: 20px; height: 20px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  color: #fff; flex-shrink: 0; z-index: 1;
  border: 2px solid var(--bg-card);
}
.timeline-dot.success { background: var(--success); }
.timeline-dot.running { background: var(--warning); animation: pulse 1.5s infinite; }
.timeline-dot.error { background: var(--danger); }
.timeline-dot.info,
.timeline-dot.warn { background: var(--primary); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

.timeline-body { flex: 1; min-width: 0; }
.timeline-agent { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.timeline-agent.has-substeps {
  cursor: pointer;
  display: flex; align-items: center; gap: 6px;
  user-select: none;
}
.timeline-agent.has-substeps:hover { color: var(--brand); }
.expand-arrow {
  flex-shrink: 0;
  color: var(--text-muted);
  transition: transform var(--transition-fast);
}
.expand-arrow.expanded { transform: rotate(90deg); }
.substep-count {
  font-size: 10px; font-weight: 500;
  color: var(--text-muted);
  background: var(--bg-input);
  padding: 1px 6px; border-radius: 8px;
  margin-left: auto;
}
.timeline-detail { font-size: 12px; color: var(--text-secondary); margin-top: 3px; line-height: 1.5; }
.timeline-sub { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }
.sub-chip {
  font-size: 11px; color: var(--text-secondary);
  background: var(--bg-input);
  border: 1px solid var(--border);
  padding: 4px 8px; border-radius: 6px;
  line-height: 1.4;
}
.timeline-time {
  font-size: 10px; color: var(--text-muted);
  font-family: var(--font-mono); margin-top: 4px;
}

/* 子步骤 */
.timeline-substeps {
  margin-top: 10px;
  padding: 8px 10px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  display: flex; flex-direction: column; gap: 6px;
}
.substep-row {
  display: flex; align-items: flex-start; gap: 8px;
  font-size: 11px;
  opacity: 0.85;
  line-height: 1.4;
}
.substep-dot {
  width: 6px; height: 6px; border-radius: 50%;
  flex-shrink: 0; margin-top: 6px;
}
.substep-dot.done { background: var(--success); }
.substep-dot.running { background: var(--warning); }
.substep-dot.error { background: var(--danger); }
.substep-row.done .substep-label { color: var(--text-primary); font-weight: 500; }
.substep-row.running .substep-label { color: var(--warning); font-weight: 600; }
.substep-row.error .substep-label { color: var(--danger); font-weight: 600; }
.substep-label { flex-shrink: 0; min-width: 80px; color: var(--text-secondary); }
.substep-detail { color: var(--text-muted); word-break: break-all; }

.log-empty {
  padding: 60px 16px; text-align: center;
  display: flex; flex-direction: column; align-items: center; gap: 8px;
}
.log-empty p { font-size: 12px; color: var(--text-muted); }

@media (max-width: 1200px) {
  .dashboard-grid { grid-template-columns: 240px 1fr; }
  .panel-right { display: none; }
}
</style>
