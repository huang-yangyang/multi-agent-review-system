/**
 * 入场动画 — 基于 IntersectionObserver，元素进入视口时触发 CSS 过渡。
 *
 * 用法 1（指令）：
 *   <div v-reveal>从下方淡入</div>
 *   <div v-reveal="'scale'">缩放淡入</div>
 *   <div v-reveal="'left'">从左侧滑入</div>
 *
 * 用法 2（交错容器）：
 *   <div v-reveal="'stagger'">
 *     <div>子项1</div>
 *     <div>子项2</div>
 *   </div>
 */
export const vReveal = {
  mounted(el, binding) {
    const type = binding.value || 'default'
    const classMap = {
      default: 'reveal',
      scale: 'reveal-scale',
      left: 'reveal-left',
      stagger: 'reveal-stagger',
    }
    el.classList.add(classMap[type] || 'reveal')

    // 移动端直接显示
    if (window.innerWidth <= 768) {
      el.classList.add('revealed')
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            el.classList.add('revealed')
            observer.unobserve(el)
          }
        })
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    )
    observer.observe(el)
    el._revealObserver = observer
  },
  unmounted(el) {
    if (el._revealObserver) el._revealObserver.disconnect()
  },
}
