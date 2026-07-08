<template>
  <el-config-provider :locale="zhCn">
    <div class="scroll-progress" aria-hidden="true"></div>

    <div class="app-container" v-if="isLoggedIn">
      <header class="app-header">
        <!-- 左：品牌 Logo -->
        <div class="header-left">
          <div class="logo" @click="$router.push('/')">
            <div class="logo-badge">
              <el-icon :size="20" color="#fff"><Connection /></el-icon>
            </div>
            <div class="logo-text">
              <div class="logo-title">LangGraph <span class="accent">Console</span></div>
              <div class="logo-sub">Multi-Agent</div>
            </div>
          </div>
        </div>

        <!-- 中：主导航 -->
        <nav class="header-nav">
          <div
            v-for="item in navItems"
            :key="item.path"
            class="nav-item"
            :class="{ active: currentPath === item.path }"
            @click="$router.push(item.path)"
          >
            <el-icon :size="16"><component :is="item.icon" /></el-icon>
            <span>{{ item.label }}</span>
          </div>
        </nav>

        <!-- 右：状态 + 用户 -->
        <div class="header-right">
          <div class="health-indicator" :title="'后端服务 ' + healthText">
            <span class="health-dot" :class="healthStatus"></span>
            <span class="health-text">{{ healthText }}</span>
          </div>

          <el-divider direction="vertical" style="border-color:var(--border);height:22px" />

          <el-dropdown trigger="click" @command="onCommand">
            <div class="user-chip">
              <div class="user-avatar">{{ avatarLetter }}</div>
              <span class="user-name">{{ username }}</span>
              <el-icon :size="12"><ArrowDown /></el-icon>
            </div>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item disabled>
                  <el-icon><User /></el-icon> {{ username }}
                </el-dropdown-item>
                <el-dropdown-item divided command="logout">
                  <el-icon><SwitchButton /></el-icon> 退出登录
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </header>

      <main class="app-main">
        <router-view v-slot="{ Component }">
          <component :is="Component" />
        </router-view>
      </main>
    </div>

    <router-view v-else />
  </el-config-provider>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import {
  Connection, DataLine, Setting,
  ArrowDown, User, SwitchButton
} from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()
const username = ref(localStorage.getItem('username') || '')

const isLoggedIn = computed(() => !!localStorage.getItem('token'))
const currentPath = computed(() => route.path)
const avatarLetter = computed(() => (username.value || 'U').charAt(0).toUpperCase())

const navItems = [
  { path: '/', label: '控制台', icon: DataLine },
  { path: '/flow', label: '流程设计', icon: Setting },
]

// —— 健康检查 ——
const healthStatus = ref('checking')   // online / offline / checking
const healthText = computed(() => ({
  online: '服务在线', offline: '服务离线', checking: '检测中'
}[healthStatus.value]))

let healthTimer = null

async function checkHealth() {
  try {
    const r = await fetch('/api/health')
    healthStatus.value = r.ok ? 'online' : 'offline'
  } catch {
    healthStatus.value = 'offline'
  }
}

// —— 用户操作 ——
function onCommand(cmd) {
  if (cmd === 'logout') handleLogout()
}

async function handleLogout() {
  try {
    await fetch('/api/auth/logout', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + localStorage.getItem('token')
      }
    })
  } catch (e) { /* ignore */ }
  localStorage.removeItem('token')
  localStorage.removeItem('username')
  router.push('/login')
}

// 捕获式监听任意滚动容器，把进度写入 --scroll-progress 供顶部进度条使用
function updateScrollProgress(e) {
  const el = (e && e.target && e.target !== document && e.target !== document.body)
    ? e.target
    : document.documentElement
  const max = el.scrollHeight - el.clientHeight
  const p = max > 0 ? Math.min(1, Math.max(0, el.scrollTop / max)) : 0
  document.documentElement.style.setProperty('--scroll-progress', p.toFixed(4))
}

onMounted(() => {
  checkHealth()
  healthTimer = setInterval(checkHealth, 30000)
  document.addEventListener('scroll', updateScrollProgress, true)
  updateScrollProgress()
})

onBeforeUnmount(() => {
  if (healthTimer) clearInterval(healthTimer)
  document.removeEventListener('scroll', updateScrollProgress, true)
})
</script>

<style scoped>
.header-nav {
  display: flex;
  align-items: center;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  transition: all var(--transition-fast);
}
.nav-item:hover {
  color: var(--text-primary);
  background: rgba(255,255,255,0.04);
}
.nav-item.active {
  color: #fff;
  background: var(--gradient-brand);
  box-shadow: var(--shadow-glow);
}

.health-indicator {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: var(--text-muted);
}
.health-dot {
  width: 8px; height: 8px; border-radius: 50%;
  transition: all var(--transition);
}
.health-dot.online {
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
  animation: pulse 2s infinite;
}
.health-dot.offline {
  background: var(--danger);
  box-shadow: 0 0 8px var(--danger);
}
.health-dot.checking {
  background: var(--warning);
  animation: pulse 1s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

.user-chip {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 10px 5px 5px;
  border-radius: var(--radius-pill);
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  cursor: pointer;
  transition: all var(--transition-fast);
  outline: none;
}
.user-chip:hover {
  border-color: var(--border-primary);
  background: var(--bg-hover);
}
.user-avatar {
  width: 26px; height: 26px; border-radius: 50%;
  background: var(--gradient-purple);
  display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 12px; font-weight: 700;
}
.user-name {
  font-size: 13px; font-weight: 500; color: var(--text-primary);
}

/* 覆盖 disabled 下拉项样式 */
:deep(.el-dropdown-menu__item.is-disabled) {
  color: var(--text-muted) !important;
  cursor: default;
}
</style>
