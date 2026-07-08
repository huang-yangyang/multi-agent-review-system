import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import 'element-plus/dist/index.css'

import App from './App.vue'
import Login from './views/Login.vue'
import Dashboard from './views/Dashboard.vue'
import AgentFlow from './views/AgentFlow.vue'
import './assets/styles/variables.css'
import './assets/styles/main.css'
import { vMagnetic } from './composables/useMagnetic'
import { vReveal } from './composables/useReveal'

const routes = [
  { path: '/login', name: 'Login', component: Login, meta: { guest: true } },
  { path: '/', name: 'Dashboard', component: Dashboard, meta: { requiresAuth: true } },
  { path: '/flow', name: 'AgentFlow', component: AgentFlow, meta: { requiresAuth: true } },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 融合：路由跳转启用 View Transitions（不支持的浏览器自动回退为普通跳转）
const _routerPush = router.push.bind(router)
router.push = (to, ...rest) => {
  if (typeof document !== 'undefined' && document.startViewTransition) {
    return document.startViewTransition(() => _routerPush(to, ...rest))
  }
  return _routerPush(to, ...rest)
}

// 路由守卫：未登录跳转到登录页
router.beforeEach((to, from, next) => {
  const token = localStorage.getItem('token')
  if (to.meta.requiresAuth && !token) {
    next('/login')
  } else if (to.meta.guest && token) {
    next('/')
  } else {
    next()
  }
})

const app = createApp(App)

for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.use(ElementPlus, { locale: zhCn })
app.use(router)
app.directive('magnetic', vMagnetic)
app.directive('reveal', vReveal)
app.mount('#app')