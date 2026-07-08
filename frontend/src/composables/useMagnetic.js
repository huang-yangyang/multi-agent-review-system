/**
 * 磁性按钮 — 鼠标靠近时元素轻微跟随，产生"被吸引"的质感。
 *
 * 用法：
 *   <button v-magnetic="{ strength: 0.3 }">按钮</button>
 *   <div v-magnetic>卡片</div>
 */
export const vMagnetic = {
  mounted(el, binding) {
    const strength = binding.value?.strength ?? 0.25
    el.classList.add('magnetic')

    const onMove = (e) => {
      const rect = el.getBoundingClientRect()
      const x = e.clientX - rect.left - rect.width / 2
      const y = e.clientY - rect.top - rect.height / 2
      el.style.transform = `translate(${x * strength}px, ${y * strength}px)`
    }

    const onLeave = () => {
      el.style.transform = 'translate(0, 0)'
    }

    el._magneticMove = onMove
    el._magneticLeave = onLeave
    el.addEventListener('mousemove', onMove)
    el.addEventListener('mouseleave', onLeave)
  },
  unmounted(el) {
    if (el._magneticMove) el.removeEventListener('mousemove', el._magneticMove)
    if (el._magneticLeave) el.removeEventListener('mouseleave', el._magneticLeave)
  },
}
