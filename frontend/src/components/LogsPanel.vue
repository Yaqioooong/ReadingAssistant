<script setup>
import { ref, nextTick, onActivated, onDeactivated } from 'vue'
import { fetchLogs } from '../api.js'

const logNames = [
  { value: 'api', label: '请求层 API' },
  { value: 'qa', label: '问答流水线 QA' },
  { value: 'ingest', label: '入库流水线 Ingest' },
  { value: 'storage', label: '存储服务 Storage' },
  { value: 'multi_agent', label: '多 Agent 实验' },
]
const logName = ref('qa')
const logLines = ref([])
const logPath = ref('')
const autoRefresh = ref(true)
const now = ref('')
const viewer = ref(null)
let timer = null

async function refreshLogs() {
  try {
    const data = await fetchLogs(logName.value, 500)
    logLines.value = data.lines || []
    logPath.value = data.path || ''
  } catch {
    logLines.value = []
  }
  await nextTick()
  stickToBottom()
}

function stickToBottom() {
  const el = viewer.value
  if (el) el.scrollTop = el.scrollHeight
}

function startTimer() {
  if (timer) clearInterval(timer)
  timer = null
  if (autoRefresh.value) {
    timer = setInterval(() => {
      now.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
      refreshLogs()
    }, 8000)
  }
}

function toggleAuto() {
  startTimer()
}

onActivated(() => {
  now.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  refreshLogs()
  startTimer()
})
onDeactivated(() => {
  if (timer) clearInterval(timer)
  timer = null
})
</script>

<template>
  <div class="logs-page">
    <header class="logs-head">
      <div class="logs-title">
        <h2>📄 运行日志</h2>
        <p class="muted logs-sub">按模块查看流水线日志，最近 500 行 · 尾随式输出</p>
      </div>
      <div class="logs-actions">
        <select v-model="logName" class="select" @change="refreshLogs">
          <option v-for="n in logNames" :key="n.value" :value="n.value">{{ n.label }}</option>
        </select>
        <button class="btn" :disabled="autoRefresh" @click="refreshLogs">手动刷新</button>
        <label class="auto-toggle" :class="{ on: autoRefresh }">
          <input v-model="autoRefresh" type="checkbox" @change="toggleAuto" />
          <span class="dot"></span>
          <span>{{ autoRefresh ? '自动刷新' : '已暂停' }}</span>
        </label>
      </div>
    </header>

    <section class="logs-body">
      <div class="logs-statusbar">
        <span class="status-chip" :class="{ live: autoRefresh }">
          <i class="status-led"></i>{{ autoRefresh ? '实时监听' : '静态视图' }}
        </span>
        <span v-if="autoRefresh" class="muted">最近刷新 {{ now }}</span>
        <span class="muted logs-file" :title="logPath">{{ logPath || '未定位日志文件' }}</span>
      </div>
      <pre ref="viewer" class="logs-viewer">
{{ logLines.join('\n') || '（暂无日志：先跑一次问答 / 入库 / 评测生成日志）' }}
      </pre>
    </section>
  </div>
</template>

<style>
.logs-page { max-width: 1040px; margin: 0 auto; padding-bottom: 40px; display: flex; flex-direction: column; gap: 14px; }
.logs-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.logs-title h2 { margin: 0 0 4px; font-size: 18px; }
.logs-sub { margin: 0; font-size: 13px; }
.logs-actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.auto-toggle { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--muted); cursor: pointer; user-select: none; }
.auto-toggle input { display: none; }
.auto-toggle .dot { width: 30px; height: 17px; border-radius: 999px; background: var(--border); position: relative; transition: background-color 0.2s; }
.auto-toggle .dot::after { content: ''; position: absolute; top: 2px; left: 2px; width: 13px; height: 13px; border-radius: 50%; background: var(--card); transition: transform 0.18s ease; }
.auto-toggle.on { color: var(--text); }
.auto-toggle.on .dot { background: var(--accent); }
.auto-toggle.on .dot::after { transform: translateX(13px); }
.logs-body { display: flex; flex-direction: column; background: var(--card); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
.logs-statusbar { display: flex; align-items: center; gap: 12px; padding: 8px 14px; border-bottom: 1px solid var(--border); font-size: 12px; background: var(--panel); }
.logs-file { margin-left: auto; max-width: 46%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.status-chip { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--muted); }
.status-chip.live { color: var(--ok); }
.status-led { width: 7px; height: 7px; border-radius: 50%; background: currentColor; animation: ledPulse 1.6s ease-in-out infinite; }
@keyframes ledPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
.logs-viewer {
  margin: 0;
  padding: 12px 16px;
  height: min(58vh, 520px);
  min-height: 260px;
  overflow: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  line-height: 1.65;
  color: var(--text);
  white-space: pre;
}
</style>
