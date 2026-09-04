<script setup>
import { ref, onActivated, onDeactivated } from 'vue'
import { fetchLogs } from '../api.js'

const logNames = [
  { value: 'api', label: 'api · 请求层' },
  { value: 'qa', label: 'qa · 问答流水线' },
  { value: 'ingest', label: 'ingest · 入库流水线' },
  { value: 'storage', label: 'storage · 存储服务' },
  { value: 'multi_agent', label: 'multi_agent · 多Agent' },
]
const logName = ref('qa')
const logLines = ref([])
const logPath = ref('')
const autoRefresh = ref(true)
const now = ref('')
let timer = null

async function refreshLogs() {
  try {
    const data = await fetchLogs(logName.value, 500)
    logLines.value = data.lines || []
    logPath.value = data.path || ''
  } catch {
    logLines.value = []
  }
}

function toggleAuto() {
  if (timer) clearInterval(timer)
  timer = null
  if (autoRefresh.value) {
    timer = setInterval(async () => {
      await refreshLogs()
      now.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    }, 8000)
  }
}

onActivated(() => {
  refreshLogs()
  now.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
  if (autoRefresh.value) {
    timer = setInterval(async () => {
      await refreshLogs()
      now.value = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    }, 8000)
  }
})
onDeactivated(() => {
  if (timer) clearInterval(timer)
  timer = null
})
</script>

<template>
  <div class="lab">
    <section class="card">
      <h2>📜 实时日志</h2>
      <div class="lab-controls">
        <select v-model="logName" class="select" @change="refreshLogs">
          <option v-for="n in logNames" :key="n.value" :value="n.value">{{ n.label }}</option>
        </select>
        <button class="btn" @click="refreshLogs">刷新</button>
        <label class="radio-inline">
          <input v-model="autoRefresh" type="checkbox" @change="toggleAuto" /> 自动刷新（8s）
        </label>
        <span v-if="autoRefresh" class="muted lab-hint">最近刷新 {{ now }}</span>
        <span class="muted lab-hint flex-1 right">{{ logPath }}</span>
      </div>
      <pre class="log-view full">{{ logLines.join('\n') || '（暂无日志，先跑一次问答/入库/评测生成日志）' }}</pre>
    </section>
  </div>
</template>

<style>
.log-view { max-height: calc(100vh - 190px); }
.lab-controls .flex-1 { flex: 1; }
.lab-controls .right { text-align: right; }
</style>
