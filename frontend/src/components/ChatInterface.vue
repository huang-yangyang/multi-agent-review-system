<template>
  <div class="chat-interface"
    @dragover.prevent="onDragOver"
    @dragleave.prevent="onDragLeave"
    @drop.prevent="onDrop"
  >
    <!-- 拖拽提示覆盖层 -->
    <transition name="fade">
      <div v-if="dragOver" class="drag-overlay">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
        </svg>
        <span>释放以添加文件</span>
      </div>
    </transition>

    <!-- 消息区 -->
    <div class="chat-messages" ref="msgContainer">
      <!-- 空状态 -->
      <div v-if="messages.length === 0" class="empty-state">
        <div class="empty-logo">
          <el-icon :size="36"><Connection /></el-icon>
        </div>
        <h3 class="empty-title">多 Agent 协作已就绪</h3>
        <p class="empty-desc">输入问题，启动智能问答流水线</p>
        <div class="empty-suggestions">
          <button
            v-for="s in suggestions"
            :key="s"
            class="suggestion-chip"
            @click="onSuggestion(s)"
          >{{ s }}</button>
        </div>
      </div>

      <!-- 消息列表 -->
      <div v-for="(msg, i) in messages" :key="i" class="msg-row" :class="msg.role">
        <!-- 用户头像 -->
        <div class="msg-avatar user-avatar" v-if="msg.role === 'user'">
          <el-icon :size="15"><User /></el-icon>
        </div>

        <div class="msg-body">
          <!-- Agent 标签 -->
          <div class="msg-head" v-if="msg.role !== 'user'">
            <span class="msg-agent">
              <el-icon :size="11"><Cpu /></el-icon>
              {{ msg.agent || 'Assistant' }}
            </span>
            <span class="msg-time" v-if="msg.time">{{ msg.time }}</span>
          </div>

          <!-- 消息气泡 -->
          <div class="msg-bubble" :class="{ 'is-error': isError(msg) }">
            <div
              v-if="msg.role !== 'user'"
              class="msg-content markdown-body"
              v-html="renderMd(msg.content)"
            ></div>
            <div v-else class="msg-content user-content">{{ msg.content }}</div>
            <!-- 流式光标 -->
            <span v-if="loading && i === messages.length - 1 && msg.role !== 'user'" class="cursor"></span>
          </div>
        </div>

        <!-- AI 头像 -->
        <div class="msg-avatar ai-avatar" v-if="msg.role !== 'user'">
          <el-icon :size="15" color="#fff"><Cpu /></el-icon>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="chat-input">
      <!-- 附件标签 -->
      <transition-group name="tag" tag="div" class="attach-tags" v-if="attachments.length">
        <span v-for="(f, i) in attachments" :key="f.name + i" class="attach-tag">
          <span class="tag-icon">
            <el-icon :size="13"><Document /></el-icon>
          </span>
          <span class="attach-name">{{ f.name }}</span>
          <button class="attach-remove" @click="removeAttachment(i)" title="移除">
            <el-icon :size="11"><Close /></el-icon>
          </button>
        </span>
      </transition-group>

      <div class="input-row">
        <div class="input-wrapper" :class="{ focused: focused }">
          <!-- 文件选择按钮 - 放在输入框内部左侧 -->
          <button class="attach-btn" @click="triggerFilePick" :title="loading ? '' : '添加附件'">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
            </svg>
          </button>
          <input
            ref="fileInput"
            type="file"
            class="file-input-hidden"
            @change="onFileSelected"
            multiple
          />

          <el-input
            v-model="input"
            :placeholder="loading ? 'AI 正在生成...' : '输入你的问题，Enter 发送'"
            @keyup.enter="handleSend"
            @keyup.escape="handleCancel"
            :disabled="loading"
            clearable
            :input-style="{ background: 'transparent', border: 'none', padding: '0' }"
            @focus="focused = true"
            @blur="focused = false"
          />
          <button
            class="send-btn"
            :class="{ 'is-cancel': loading, 'has-text': input.length > 0 }"
            @click="loading ? handleCancel() : handleSend()"
            :disabled="!loading && !input.trim()"
            :title="loading ? '停止生成' : '发送消息'"
          >
            <el-icon :size="17">
              <component :is="loading ? VideoPause : Promotion" />
            </el-icon>
          </button>
        </div>
      </div>
      <p class="input-hint">支持上传文档附件，Agent 将读取文件内容辅助回答</p>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'
import {
  User, Cpu, Promotion, VideoPause, Connection, Plus, Document, Close
} from '@element-plus/icons-vue'
import { renderMarkdown, bindCopyButtons } from '../utils/markdown'

const props = defineProps({
  loading: Boolean,
  messages: { type: Array, default: () => [] }
})

const emit = defineEmits(['send', 'cancel'])
const input = ref('')
const msgContainer = ref(null)
const focused = ref(false)
const fileInput = ref(null)
const attachments = ref([])
const dragOver = ref(false)

const suggestions = [
  '什么是大语言模型？',
  'RAG 检索增强生成原理',
  '解释一下 ReAct 框架'
]

function renderMd(text) {
  return renderMarkdown(text || '')
}

function isError(msg) {
  return msg.role !== 'user' && /失败|异常|错误|error/i.test(msg.content || '')
}

function onSuggestion(text) {
  if (props.loading) return
  emit('send', text, [])
}

function triggerFilePick() {
  fileInput.value?.click()
}

function onFileSelected(e) {
  const files = e.target.files
  if (files) addFiles(files)
  fileInput.value.value = ''
}

function addFiles(fileList) {
  for (const f of fileList) {
    attachments.value.push({
      name: f.name,
      file: f,
      size: f.size,
    })
  }
}

function onDragOver() {
  dragOver.value = true
}

function onDragLeave(e) {
  // 仅在真正离开容器时隐藏
  if (e.currentTarget === e.target) {
    dragOver.value = false
  }
}

function onDrop(e) {
  dragOver.value = false
  const files = e.dataTransfer?.files
  if (files && files.length) addFiles(files)
}

function removeAttachment(i) {
  attachments.value.splice(i, 1)
}

async function handleSend() {
  const text = input.value.trim()
  if (!text || props.loading) return

  // 读取附件：文本文件用 readAsText，二进制文件(PDF/DOCX)用 readAsDataURL
  const files = []
  const BINARY_EXTS = ['.pdf', '.docx', '.doc', '.pptx', '.xlsx']
  for (const a of attachments.value) {
    if (a.file) {
      const ext = '.' + a.name.split('.').pop().toLowerCase()
      const isBinary = BINARY_EXTS.includes(ext)
      const content = await new Promise((resolve) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result)
        reader.onerror = () => resolve(null)
        if (isBinary) {
          reader.readAsDataURL(a.file)  // PDF/DOCX → base64
        } else {
          reader.readAsText(a.file)      // MD/TXT → 文本
        }
      })
      files.push({ name: a.name, content: content || '', encoding: isBinary ? 'base64' : 'text' })
    }
  }

  emit('send', text, files)
  input.value = ''
  attachments.value = []
}

function handleCancel() {
  if (!props.loading) return
  emit('cancel')
}

function scrollToBottom() {
  nextTick(() => {
    if (msgContainer.value) {
      msgContainer.value.scrollTop = msgContainer.value.scrollHeight
      // 绑定新生成的代码块复制按钮
      bindCopyButtons(msgContainer.value)
    }
  })
}

watch(() => props.messages.length, scrollToBottom)
// 内容更新也滚动（流式追加）
watch(
  () => props.messages.map(m => m.content).join('|'),
  scrollToBottom
)
</script>

<style scoped>
.chat-interface {
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
}

/* 拖拽覆盖层 */
.drag-overlay {
  position: absolute;
  inset: 0;
  z-index: 10;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: rgba(var(--primary-rgb), 0.06);
  backdrop-filter: blur(4px);
  border: 2px dashed var(--border-primary);
  border-radius: 16px;
  color: var(--secondary);
  font-size: 15px;
  font-weight: 600;
  pointer-events: none;
}
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 8px 4px 16px;
}

/* ===== 空状态 ===== */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  gap: 10px;
}
.empty-logo {
  width: 72px; height: 72px; border-radius: 18px;
  background: var(--gradient-brand-soft);
  border: 1px solid var(--border-primary);
  display: flex; align-items: center; justify-content: center;
  color: var(--secondary);
  margin-bottom: 8px;
  box-shadow: var(--shadow-glow);
}
.empty-title { font-size: 18px; font-weight: 700; color: var(--text-primary); }
.empty-desc { font-size: 13px; color: var(--text-muted); }
.empty-suggestions {
  display: flex; flex-wrap: wrap; gap: 8px;
  justify-content: center; margin-top: 16px; max-width: 480px;
}
.suggestion-chip {
  padding: 7px 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-pill);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all var(--transition-fast);
}
.suggestion-chip:hover {
  border-color: var(--border-primary);
  color: var(--secondary);
  background: var(--primary-dim);
  transform: translateY(-1px);
}

/* ===== 消息行 ===== */
.msg-row {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  animation: msgIn 0.4s var(--ease-out) backwards;
}
.msg-row.user { flex-direction: row-reverse; }

@keyframes msgIn {
  from { opacity: 0; transform: translateY(14px) scale(0.97); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

.msg-avatar {
  width: 32px; height: 32px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.msg-avatar.ai-avatar {
  background: var(--gradient-purple);
  box-shadow: 0 2px 8px rgba(139,92,246,0.3);
}
.msg-avatar.user-avatar {
  background: var(--gradient-brand);
  color: #fff;
  box-shadow: 0 2px 8px rgba(var(--primary-rgb),0.3);
}

.msg-body { max-width: 78%; display: flex; flex-direction: column; }
.msg-row.user .msg-body { align-items: flex-end; }

.msg-head {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 5px; padding: 0 4px;
}
.msg-agent {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 11px; font-weight: 600; color: var(--secondary);
}
.msg-time {
  font-size: 11px; color: var(--text-muted);
  font-family: var(--font-mono);
}

.msg-bubble {
  padding: 14px 18px;
  border-radius: 14px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  position: relative;
  max-width: 100%;
  transition: border-color .3s var(--ease), box-shadow .3s var(--ease);
}
.msg-bubble:hover {
  border-color: var(--border-strong);
  box-shadow: 0 2px 16px rgba(0,0,0,.15);
}
.msg-row.user .msg-bubble {
  background: var(--gradient-brand-soft);
  border-color: var(--border-primary);
}
.msg-row.user .msg-bubble:hover {
  box-shadow: var(--shadow-glow);
}
.msg-bubble.is-error {
  border-color: rgba(239,68,68,0.4);
  background: rgba(239,68,68,0.06);
}
.user-content {
  font-size: 14px; line-height: 1.65;
  white-space: pre-wrap; word-break: break-word;
}

/* 流式光标 */
.cursor {
  display: inline-block;
  width: 8px; height: 16px;
  background: var(--secondary);
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 1s step-end infinite;
  border-radius: 1px;
}
@keyframes blink { 0%,50%{opacity:1} 51%,100%{opacity:0} }

/* ===== 输入区 ===== */
.chat-input {
  padding-top: 14px;
  border-top: 1px solid var(--border);
}

/* 附件标签 */
.attach-tags {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin-bottom: 10px;
}
.tag-enter-active, .tag-leave-active {
  transition: all 0.25s ease;
}
.tag-enter-from, .tag-leave-to {
  opacity: 0;
  transform: scale(0.85);
}
.attach-tag {
  display: inline-flex; align-items: center; gap: 0;
  height: 28px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 12px;
  color: var(--text-secondary);
  overflow: hidden;
  transition: all 0.2s ease;
}
.attach-tag:hover {
  border-color: var(--border-primary);
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.tag-icon {
  display: flex; align-items: center; justify-content: center;
  width: 28px; height: 28px;
  background: var(--primary-dim);
  color: var(--secondary);
  flex-shrink: 0;
}
.attach-name {
  padding: 0 8px;
  max-width: 180px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
}
.attach-remove {
  display: flex; align-items: center; justify-content: center;
  width: 22px; height: 22px;
  margin-right: 3px;
  border: none; border-radius: 6px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0;
  transition: all 0.15s ease;
}
.attach-remove:hover {
  background: rgba(239,68,68,0.12);
  color: #ef4444;
}

/* 输入行 */
.input-row {
  display: flex; gap: 0; align-items: center;
}
.file-input-hidden { display: none; }

.input-wrapper {
  display: flex; gap: 6px; align-items: center;
  padding: 5px 5px 5px 8px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: 14px;
  transition: all 0.2s ease;
  flex: 1;
}
.input-wrapper.focused {
  border-color: var(--border-primary);
  box-shadow: 0 0 0 3px rgba(var(--primary-rgb), 0.08);
  background: var(--bg-card);
}
.input-wrapper:hover {
  border-color: var(--border-strong);
}

/* 附件按钮 - 输入框内左侧 */
.attach-btn {
  display: flex; align-items: center; justify-content: center;
  width: 34px; height: 34px;
  border: none; border-radius: 10px;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.2s ease;
}
.attach-btn:hover {
  background: var(--primary-dim);
  color: var(--secondary);
}
.attach-btn:active {
  transform: scale(0.93);
}

.input-wrapper :deep(.el-input__wrapper) {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  flex: 1;
  padding-left: 2px !important;
}
.input-wrapper :deep(.el-input__inner) {
  font-size: 13.5px;
  color: var(--text-primary);
}
.input-wrapper :deep(.el-input__wrapper.is-focus) {
  box-shadow: none !important;
}

.send-btn {
  display: flex; align-items: center; justify-content: center;
  width: 34px; height: 34px;
  border: none; border-radius: 10px;
  background: var(--gradient-brand);
  color: #fff;
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.2s ease;
}
.send-btn:hover:not(:disabled) {
  filter: brightness(1.15);
  box-shadow: 0 2px 8px rgba(var(--primary-rgb), 0.35);
  transform: scale(1.04);
}
.send-btn:active:not(:disabled) {
  transform: scale(0.95);
}
.send-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}
.send-btn.is-cancel {
  background: linear-gradient(135deg, #ef4444, #f59e0b);
  width: auto;
  padding: 0 12px;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
}
.send-btn.is-cancel:hover {
  filter: brightness(1.15);
  box-shadow: 0 2px 8px rgba(239,68,68,0.35);
}

.input-hint {
  text-align: center;
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 8px;
  opacity: 0.6;
}
</style>

<style>
/* ===== Markdown 渲染样式（非 scoped，作用于 v-html）===== */
.markdown-body { font-size: 14px; line-height: 1.7; color: var(--text-primary); }
.markdown-body .md-p { margin: 0 0 10px; }
.markdown-body .md-p:last-child { margin-bottom: 0; }
.markdown-body .md-h { font-weight: 700; line-height: 1.3; margin: 14px 0 8px; }
.markdown-body .md-h1 { font-size: 20px; }
.markdown-body .md-h2 { font-size: 17px; }
.markdown-body .md-h3 { font-size: 15px; }
.markdown-body .md-h4,
.markdown-body .md-h5,
.markdown-body .md-h6 { font-size: 14px; color: var(--text-secondary); }

.markdown-body strong { color: var(--text-primary); font-weight: 700; }
.markdown-body em { font-style: italic; color: var(--text-secondary); }
.markdown-body del { color: var(--text-muted); }
.markdown-body a { color: var(--secondary); text-decoration: none; border-bottom: 1px dashed currentColor; }
.markdown-body a:hover { opacity: 0.8; }

.markdown-body .md-inline-code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: rgba(var(--secondary-rgb), 0.12);
  color: var(--secondary);
  padding: 1px 6px;
  border-radius: 5px;
}

.markdown-body .md-ul,
.markdown-body .md-ol { margin: 6px 0 10px; padding-left: 22px; }
.markdown-body .md-ul li { list-style: disc; margin-bottom: 3px; }
.markdown-body .md-ol li { list-style: decimal; margin-bottom: 3px; }

.markdown-body .md-quote {
  border-left: 3px solid var(--primary);
  padding: 4px 12px;
  margin: 8px 0;
  background: rgba(var(--primary-rgb), 0.06);
  color: var(--text-secondary);
  border-radius: 0 6px 6px 0;
}

.markdown-body .md-hr {
  border: none;
  border-top: 1px solid var(--border-strong);
  margin: 14px 0;
}

.markdown-body .md-table {
  width: 100%;
  border-collapse: collapse;
  margin: 8px 0 12px;
  font-size: 13px;
  overflow: hidden;
  border-radius: var(--radius-sm);
}
.markdown-body .md-table th,
.markdown-body .md-table td {
  border: 1px solid var(--border);
  padding: 8px 12px;
  text-align: left;
}
.markdown-body .md-table th {
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-weight: 600;
}

.markdown-body .md-code-block {
  background: var(--bg-input);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin: 10px 0 12px;
  overflow: hidden;
}
.markdown-body .md-code-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px;
  background: rgba(255,255,255,0.03);
  border-bottom: 1px solid var(--border);
}
.markdown-body .md-code-lang {
  font-size: 11px; color: var(--text-muted);
  font-family: var(--font-mono); text-transform: lowercase;
}
.markdown-body .md-code-copy {
  background: transparent;
  border: 1px solid var(--border-strong);
  color: var(--text-muted);
  font-size: 11px;
  padding: 2px 10px;
  border-radius: 5px;
  cursor: pointer;
  transition: all 0.15s;
}
.markdown-body .md-code-copy:hover {
  color: var(--secondary);
  border-color: var(--secondary);
}
.markdown-body .md-code-copy.copied {
  color: var(--success);
  border-color: var(--success);
}
.markdown-body .md-code-block code {
  display: block;
  padding: 12px 14px;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.6;
  color: var(--text-secondary);
  overflow-x: auto;
  white-space: pre;
}
</style>
