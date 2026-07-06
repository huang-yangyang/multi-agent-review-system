/**
 * 轻量 Markdown 渲染器（零依赖）
 * 支持：代码块 ```、行内代码 `、标题 #、无序/有序列表、
 *      引用 >、表格 | |、加粗 **、斜体 *、链接 []()、分隔线、转义防 XSS
 * 输出 HTML 字符串，由组件以 v-html 渲染（调用方需自行控制源可信）
 */

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/** 行内格式：加粗、斜体、行内代码、链接 */
function renderInline(text) {
  let s = escapeHtml(text)
  // 行内代码（优先处理，避免内部被其它规则破坏）
  s = s.replace(/`([^`]+?)`/g, (_, c) => `<code class="md-inline-code">${c}</code>`)
  // 链接 [text](url)
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, t, u) =>
    `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`)
  // 加粗 **text** 或 __text__
  s = s.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>')
  s = s.replace(/__([^_]+?)__/g, '<strong>$1</strong>')
  // 斜体 *text* 或 _text_（避免与加粗冲突，要求紧贴字符）
  s = s.replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>')
  // 删除线 ~~text~~
  s = s.replace(/~~([^~]+?)~~/g, '<del>$1</del>')
  return s
}

export function renderMarkdown(src) {
  if (!src) return ''
  const lines = String(src).replace(/\r\n/g, '\n').split('\n')
  const html = []
  let i = 0

  while (i < lines.length) {
    let line = lines[i]

    // —— 代码块 ``` ——
    if (/^```/.test(line.trim())) {
      const lang = line.trim().slice(3).trim()
      const buf = []
      i++
      while (i < lines.length && !/^```/.test(lines[i].trim())) {
        buf.push(lines[i])
        i++
      }
      i++ // 跳过结束的 ```
      const code = escapeHtml(buf.join('\n'))
      const langAttr = lang ? ` data-lang="${escapeHtml(lang)}"` : ''
      html.push(
        `<pre class="md-code-block"><div class="md-code-head"><span class="md-code-lang">${escapeHtml(lang || 'code')}</span><button class="md-code-copy" data-code="${encodeURIComponent(buf.join('\n'))}">复制</button></div><code${langAttr}>${code}</code></pre>`
      )
      continue
    }

    // —— 空行 ——
    if (line.trim() === '') {
      i++
      continue
    }

    // —— 分隔线 ——
    if (/^(\s*[-*_]){3,}\s*$/.test(line)) {
      html.push('<hr class="md-hr" />')
      i++
      continue
    }

    // —— 标题 # ~ ###### ——
    const h = line.match(/^(#{1,6})\s+(.*)$/)
    if (h) {
      const level = h[1].length
      html.push(`<h${level} class="md-h md-h${level}">${renderInline(h[2])}</h${level}>`)
      i++
      continue
    }

    // —— 引用 > ——
    if (/^>\s?/.test(line)) {
      const buf = []
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^>\s?/, ''))
        i++
      }
      html.push(`<blockquote class="md-quote">${renderInline(buf.join(' '))}</blockquote>`)
      continue
    }

    // —— 表格 | a | b | + 分隔行 ——
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length &&
        /^\s*\|?[\s:|-]+\|[\s:|-]+\s*$/.test(lines[i + 1])) {
      const header = line.split('|').slice(1, -1).map(c => c.trim())
      i += 2 // 跳过分隔行
      const rows = []
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        rows.push(lines[i].split('|').slice(1, -1).map(c => c.trim()))
        i++
      }
      let table = '<table class="md-table"><thead><tr>'
      header.forEach(c => { table += `<th>${renderInline(c)}</th>` })
      table += '</tr></thead><tbody>'
      rows.forEach(r => {
        table += '<tr>'
        r.forEach(c => { table += `<td>${renderInline(c)}</td>` })
        table += '</tr>'
      })
      table += '</tbody></table>'
      html.push(table)
      continue
    }

    // —— 无序列表 - / * / + ——
    if (/^\s*([-*+])\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*([-*+])\s+/.test(lines[i])) {
        items.push(`<li>${renderInline(lines[i].replace(/^\s*([-*+])\s+/, ''))}</li>`)
        i++
      }
      html.push(`<ul class="md-ul">${items.join('')}</ul>`)
      continue
    }

    // —— 有序列表 1. ——
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(`<li>${renderInline(lines[i].replace(/^\s*\d+\.\s+/, ''))}</li>`)
        i++
      }
      html.push(`<ol class="md-ol">${items.join('')}</ol>`)
      continue
    }

    // —— 普通段落（合并连续非空行）——
    const para = []
    while (i < lines.length && lines[i].trim() !== '' &&
           !/^```/.test(lines[i].trim()) && !/^(#{1,6})\s/.test(lines[i]) &&
           !/^\s*([-*+])\s+/.test(lines[i]) && !/^\s*\d+\.\s+/.test(lines[i]) &&
           !/^>\s?/.test(lines[i]) && !/^\s*\|.*\|\s*$/.test(lines[i]) &&
           !/^(\s*[-*_]){3,}\s*$/.test(lines[i])) {
      para.push(lines[i])
      i++
    }
    html.push(`<p class="md-p">${renderInline(para.join(' '))}</p>`)
  }

  return html.join('\n')
}

/**
 * 绑定代码块复制按钮（在 v-html 渲染后调用）
 * @param {HTMLElement} root 容器元素
 */
export function bindCopyButtons(root) {
  if (!root) return
  root.querySelectorAll('.md-code-copy').forEach(btn => {
    if (btn.__mdBound) return
    btn.__mdBound = true
    btn.addEventListener('click', () => {
      const code = decodeURIComponent(btn.dataset.code || '')
      navigator.clipboard.writeText(code).then(() => {
        const old = btn.textContent
        btn.textContent = '已复制'
        btn.classList.add('copied')
        setTimeout(() => {
          btn.textContent = old
          btn.classList.remove('copied')
        }, 1500)
      }).catch(() => {})
    })
  })
}
