<template>
  <div class="login-page">
    <!-- 左侧：品牌介绍区 -->
    <aside class="brand-panel" v-reveal="'left'">
      <div class="brand-bg-orb orb-1"></div>
      <div class="brand-bg-orb orb-2"></div>
      <div class="brand-grid"></div>

      <div class="brand-content">
        <div class="brand-logo">
          <div class="brand-logo-badge">
            <el-icon :size="22"><Connection /></el-icon>
          </div>
          <div class="brand-logo-text">
            <h1>LangGraph <span class="grad">Console</span></h1>
            <p>MULTI-AGENT INTELLIGENCE PLATFORM</p>
          </div>
        </div>

        <div class="brand-headline">
          <h2>多智能体协作<br /><span class="grad">智能问答引擎</span></h2>
          <p>问题拆解 · 模型路由 · 人机协同 · ReAct 推理</p>
        </div>

        <div class="brand-features">
          <div class="feature-item" v-for="f in features" :key="f.title">
            <div class="feature-icon" :style="{ background: f.gradient }">
              <el-icon :size="16"><component :is="f.icon" /></el-icon>
            </div>
            <div class="feature-text">
              <div class="feature-title">{{ f.title }}</div>
              <div class="feature-desc">{{ f.desc }}</div>
            </div>
          </div>
        </div>

        <div class="brand-footer">
          <span class="dot online"></span> 系统在线 · 基于 LangGraph 构建
        </div>
      </div>
    </aside>

    <!-- 右侧：登录卡 -->
    <main class="login-panel">
      <div class="login-card aurora" v-reveal="'scale'">
        <div class="login-header">
          <h2>欢迎回来</h2>
          <p>登录以进入控制台</p>
        </div>

        <el-form
          ref="formRef"
          :model="form"
          :rules="rules"
          class="login-form"
          label-position="top"
          @keyup.enter="handleLogin"
        >
          <el-form-item prop="username" label="用户名">
            <el-input
              v-model="form.username"
              placeholder="请输入用户名"
              size="large"
              :prefix-icon="User"
              clearable
            />
          </el-form-item>
          <el-form-item prop="password" label="密码">
            <el-input
              v-model="form.password"
              type="password"
              placeholder="请输入密码"
              size="large"
              :prefix-icon="Lock"
              show-password
            />
          </el-form-item>
          <el-form-item>
            <el-button
              type="primary"
              size="large"
              class="login-btn"
              v-magnetic
              :loading="loading"
              @click="handleLogin"
            >
              <el-icon v-if="!loading" style="margin-right:6px"><Right /></el-icon>
              {{ loading ? '登录中...' : '登 录' }}
            </el-button>
          </el-form-item>
          <transition name="fade">
            <p v-if="error" class="error-msg">
              <el-icon><WarningFilled /></el-icon> {{ error }}
            </p>
          </transition>
        </el-form>

        <div class="quick-login">
          <div class="quick-title"><span>快捷登录（密码均为 1234）</span></div>
          <div class="quick-chips">
            <button
              v-for="acc in presets"
              :key="acc.username"
              type="button"
              class="quick-chip"
              :class="{ active: form.username === acc.username }"
              @click="fillAccount(acc)"
            >
              <el-icon :size="14" :style="{ color: acc.color }"><component :is="acc.icon" /></el-icon>
              <span class="chip-name">{{ acc.username }}</span>
              <span class="chip-role">{{ acc.role }}</span>
            </button>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import {
  User, Lock, Right, Connection, WarningFilled,
  Cpu, DataAnalysis, ChatDotRound, Files
} from '@element-plus/icons-vue'

const router = useRouter()
const formRef = ref(null)
const loading = ref(false)
const error = ref('')

const form = reactive({ username: '', password: '' })

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const presets = [
  { username: 'admin',       role: '系统管理员', color: '#22d3ee', icon: Cpu },
  { username: 'legal_lead',  role: '法律主管',   color: '#a78bfa', icon: ChatDotRound },
  { username: 'legal_user',  role: '法律员工',   color: '#818cf8', icon: ChatDotRound },
  { username: 'hr_lead',     role: '人事主管',   color: '#f59e0b', icon: Files },
  { username: 'hr_user',     role: '人事员工',   color: '#fbbf24', icon: Files },
  { username: 'general_user',role: '员工',       color: '#34d399', icon: Files },
]

const features = [
  { title: '智能拆解', desc: '复杂问题自动分解为子问题', icon: DataAnalysis, gradient: 'linear-gradient(135deg,#8b5cf6,#6366f1)' },
  { title: 'ReAct 推理', desc: 'LLM 自主调用工具循环决策', icon: Cpu, gradient: 'linear-gradient(135deg,#22d3ee,#0ea5e9)' },
  { title: '混合检索', desc: 'Embedding + BM25 + RRF 知识库', icon: Files, gradient: 'linear-gradient(135deg,#f59e0b,#ef4444)' }
]

function fillAccount(acc) {
  form.username = acc.username
  form.password = '1234'
  error.value = ''
}

async function handleLogin() {
  if (!formRef.value) return
  const valid = await formRef.value.validate().catch(() => false)
  if (!valid) return

  loading.value = true
  error.value = ''

  try {
    const r = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: form.username, password: form.password })
    })
    const data = await r.json()
    if (r.ok && data.token) {
      localStorage.setItem('token', data.token)
      localStorage.setItem('username', data.username)
      window.location.href = '/'
    } else {
      error.value = data.detail || data.error || '登录失败'
    }
  } catch (e) {
    error.value = '网络错误，请检查后端服务是否启动'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  display: grid;
  grid-template-columns: 1.1fr 1fr;
  height: 100vh;
  position: relative;
  z-index: 1;
}

/* ===== 左侧品牌区 ===== */
.brand-panel {
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(ellipse at top left, rgba(99,102,241,0.18), transparent 50%),
    radial-gradient(ellipse at bottom right, rgba(34,211,238,0.14), transparent 50%),
    linear-gradient(160deg, #0c1120 0%, #0a0e17 100%);
  display: flex;
  align-items: center;
  padding: 56px 64px;
}

.brand-bg-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.5;
  animation: float 8s ease-in-out infinite;
}
.orb-1 { width: 320px; height: 320px; background: #6366f1; top: -80px; left: -60px; }
.orb-2 { width: 280px; height: 280px; background: #22d3ee; bottom: -60px; right: -40px; animation-delay: -4s; }

@keyframes float {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(20px, -20px); }
}

.brand-grid {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(148,163,220,0.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,220,0.05) 1px, transparent 1px);
  background-size: 40px 40px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
}

.brand-content {
  position: relative;
  z-index: 2;
  width: 100%;
}

.brand-logo {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 64px;
}
.brand-logo-badge {
  width: 48px; height: 48px;
  border-radius: 12px;
  background: var(--gradient-brand);
  display: flex; align-items: center; justify-content: center;
  color: #fff;
  box-shadow: var(--shadow-glow);
}
.brand-logo-text h1 {
  font-size: 20px; font-weight: 800; letter-spacing: -0.5px;
}
.brand-logo-text h1 .grad {
  background: var(--gradient-text);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 400;
}
.brand-logo-text p {
  font-size: 10px; color: var(--text-muted); letter-spacing: 1.5px; margin-top: 3px;
}

.brand-headline {
  margin-bottom: 48px;
}
.brand-headline h2 {
  font-size: var(--step-2); font-weight: 800; line-height: 1.25; letter-spacing: -1px;
  margin-bottom: 16px;
}
.brand-headline h2 .grad {
  background: var(--gradient-text);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.brand-headline p {
  font-size: 14px; color: var(--text-secondary); letter-spacing: 0.3px;
}

.brand-features {
  display: flex; flex-direction: column; gap: 18px;
  margin-bottom: 48px;
}
.feature-item {
  display: flex; align-items: center; gap: 14px;
  padding: 14px 18px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius);
  backdrop-filter: blur(8px);
  transition: all var(--transition);
}
.feature-item:hover {
  transform: translateX(4px);
  background: rgba(255,255,255,0.06);
  border-color: var(--border-primary);
}
.feature-icon {
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  color: #fff; flex-shrink: 0;
}
.feature-title { font-size: 14px; font-weight: 600; }
.feature-desc { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

.brand-footer {
  font-size: 12px; color: var(--text-muted);
  display: flex; align-items: center; gap: 8px;
}
.dot.online {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ===== 右侧登录区 ===== */
.login-panel {
  display: flex; align-items: center; justify-content: center;
  background: var(--bg-primary);
  padding: 40px;
}
.login-card {
  width: 100%; max-width: 400px;
}
/* 融合：极光描边登录卡（套用全局 .aurora 旋转渐变边框） */
.login-card.aurora {
  background: var(--bg-card);
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow-lg);
  transition: box-shadow var(--transition);
}
/* 融合：输入框聚焦时整卡微光（:has() 父级响应） */
.login-card.aurora:has(.el-input__wrapper.is-focus) {
  box-shadow: var(--shadow-lg), 0 0 0 3px rgba(var(--primary-rgb), 0.14);
}
.login-header { margin-bottom: 32px; }
.login-header h2 {
  font-size: var(--step-1); font-weight: 700; letter-spacing: -0.5px;
}
.login-header p {
  font-size: 13px; color: var(--text-muted); margin-top: 6px;
}

.login-form :deep(.el-form-item__label) {
  color: var(--text-secondary) !important;
  font-size: 13px; font-weight: 500;
  padding-bottom: 6px;
}

.login-btn {
  width: 100%;
  font-weight: 600; font-size: 15px;
  height: 44px;
}

.error-msg {
  display: flex; align-items: center; gap: 6px;
  color: var(--danger); font-size: 13px; text-align: center;
  justify-content: center;
}

/* ===== 快捷登录 ===== */
.quick-login { margin-top: 28px; }
.quick-title {
  text-align: center; font-size: 12px; color: var(--text-muted);
  margin-bottom: 14px; position: relative;
}
.quick-title::before, .quick-title::after {
  content: ''; position: absolute; top: 50%; width: 30%; height: 1px;
  background: var(--border);
}
.quick-title::before { left: 0; }
.quick-title::after { right: 0; }

/* 融合：Subgrid 让各 chip 的「图标 / 名称 / 角色」跨卡片对齐 */
.quick-chips {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
  grid-template-rows: auto auto auto;
  gap: 8px;
}
.quick-chip {
  grid-row: span 3;
  display: grid;
  grid-template-rows: subgrid;
  align-items: center;
  padding: 12px 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all var(--transition);
  color: var(--text-secondary);
}
.quick-chip:hover {
  border-color: var(--border-primary);
  background: var(--bg-hover);
  transform: translateY(-2px);
}
.quick-chip.active {
  border-color: var(--primary);
  background: var(--primary-dim);
  box-shadow: var(--shadow-glow);
}
.chip-name { font-size: 13px; font-weight: 600; color: var(--text-primary); }
.chip-role { font-size: 10px; color: var(--text-muted); }

/* 响应式：窄屏隐藏左侧 */
@media (max-width: 900px) {
  .login-page { grid-template-columns: 1fr; }
  .brand-panel { display: none; }
}
</style>
