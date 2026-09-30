<script setup>
import { computed, ref } from 'vue'
import { runEval, runMultiAgent } from '../api.js'

const evalMode = ref('fake')
const evalSource = ref('golden')
const evalRunning = ref(false)
const evalReport = ref(null)
const uploadInput = ref(null)
const datasetFile = ref(null)
const datasetError = ref('')
const datasetCases = ref([])
const dragover = ref(false)
const selectedMetrics = ref(['answerable_accuracy', 'citation_coverage', 'latency'])
const expandedResult = ref(null)
const metricOptions = [
  { key: 'answerable_accuracy', label: '可答准确率', hint: '关键词命中' },
  { key: 'unanswerable_recognition', label: '不可答识别', hint: '拒答判断' },
  { key: 'citation_coverage', label: '引用覆盖率', hint: '答案可追溯' },
  { key: 'latency', label: '响应延迟', hint: '平均耗时' },
]
const customCasesText = ref('张三喜欢谁？ => 李四\n王五喜欢李四吗？ => 不|没|否\n少白公的原名叫什么？')
const hasDataset = computed(() => evalSource.value === 'upload' && datasetCases.value.length > 0)
const activeCases = computed(() => evalSource.value === 'upload' ? datasetCases.value : evalSource.value === 'custom' ? parseCustomCases(customCasesText.value) : [])

function parseCustomCases(text) {
  return text.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
    const [question, keywords] = line.split('=>').map((part) => part.trim())
    return { question, expect_keywords: keywords ? keywords.split('|').map((s) => s.trim()).filter(Boolean) : [] }
  })
}
function parseDataset(text, filename) {
  const isCsv = filename.toLowerCase().endsWith('.csv')
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter((line) => line.trim())
  if (isCsv) {
    const headers = lines.shift().split(',').map((s) => s.trim())
    const questionIndex = headers.indexOf('question')
    if (questionIndex < 0) throw new Error('CSV 必须包含 question 列')
    return lines.map((line) => {
      const cells = line.split(',').map((s) => s.trim())
      const keywordIndex = headers.indexOf('expect_keywords'); const unanswerableIndex = headers.indexOf('expect_unanswerable')
      return { question: cells[questionIndex], expect_keywords: (cells[keywordIndex] || '').split('|').filter(Boolean), expect_unanswerable: cells[unanswerableIndex] === 'true' }
    }).filter((item) => item.question)
  }
  return lines.map((line, index) => {
    let item
    try { item = JSON.parse(line) } catch { throw new Error(`第 ${index + 1} 行不是有效 JSON`) }
    if (!item.question || typeof item.question !== 'string') throw new Error(`第 ${index + 1} 行缺少 question`)
    return { question: item.question, document: item.document || null, expect_keywords: item.expect_keywords || [], expect_unanswerable: Boolean(item.expect_unanswerable) }
  })
}
function handleFile(file) {
  if (!file) return
  datasetError.value = ''
  if (!/\.(jsonl|ndjson|csv)$/i.test(file.name)) { datasetError.value = '仅支持 .jsonl、.ndjson 或 .csv 文件'; return }
  const reader = new FileReader()
  reader.onload = () => { try { const cases = parseDataset(String(reader.result), file.name); if (!cases.length) throw new Error('文件中没有可用样本'); datasetFile.value = file; datasetCases.value = cases; evalSource.value = 'upload' } catch (error) { datasetError.value = error.message; datasetFile.value = null; datasetCases.value = [] } }
  reader.readAsText(file)
}
function onDrop(event) { dragover.value = false; handleFile(event.dataTransfer.files?.[0]) }
function downloadTemplate(type) {
  const content = type === 'csv' ? 'question,document,expect_keywords,expect_unanswerable\n张三喜欢谁？,,李四,false\n这个文档有没有提到火星？,,,true\n' : '{"question":"张三喜欢谁？","document":null,"expect_keywords":["李四"],"expect_unanswerable":false}\n{"question":"这个文档有没有提到火星？","document":null,"expect_keywords":[],"expect_unanswerable":true}\n'
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' }); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `reading-assistant-eval-template.${type === 'csv' ? 'csv' : 'jsonl'}`; link.click(); URL.revokeObjectURL(link.href)
}
function metricOf(summary, key) { const value = summary?.[key]; return value === null || value === undefined ? '—' : key === 'latency' ? `${value} ms` : `${(value * 100).toFixed(0)}%` }
function toggleMetric(key) { selectedMetrics.value = selectedMetrics.value.includes(key) ? selectedMetrics.value.filter((item) => item !== key) : [...selectedMetrics.value, key] }
async function startEval() { if (evalRunning.value || (evalSource.value === 'upload' && !hasDataset.value)) return; evalRunning.value = true; evalReport.value = null; expandedResult.value = null; try { evalReport.value = await runEval(evalMode.value, 0, activeCases.value) } catch (error) { evalReport.value = { error: error.message } } finally { evalRunning.value = false } }
function exportReport() { if (!evalReport.value || evalReport.value.error) return; const blob = new Blob([JSON.stringify(evalReport.value, null, 2)], { type: 'application/json' }); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'reading-assistant-eval-report.json'; link.click(); URL.revokeObjectURL(link.href) }

const presetQuestions = ['张三喜欢谁？', '罗辑在第三章放下了什么？', '少白公的结局是什么？']
const agentQuestion = ref(presetQuestions[0]); const agentRunning = ref(false); const agentResult = ref(null)
async function runAgent() { if (!agentQuestion.value.trim() || agentRunning.value) return; agentRunning.value = true; agentResult.value = null; try { agentResult.value = await runMultiAgent(agentQuestion.value) } catch (error) { agentResult.value = { error: error.message } } finally { agentRunning.value = false } }
</script>

<template>
  <div class="lab eval-workbench">
    <section class="lab-hero eval-hero"><div class="eval-hero-copy"><div class="lab-hero-badges"><span class="tag-chip experimental">Evaluation workspace</span><span class="tag-chip">与生产链路同源</span></div><h1 class="lab-hero-title">评测中心 <span>·</span> 把回答质量变成可比较的数据</h1><p class="lab-hero-desc">上传一份属于你的问题集，选择想看的指标，跑一次真实问答链路。结果支持逐题复盘、失败定位和报告导出。</p></div><div class="eval-hero-status"><span class="status-dot"></span><span>评测服务就绪</span><small>API · RAG · 引用链路</small></div></section>

    <section class="eval-grid"><div class="card eval-setup-card"><div class="section-kicker">01 / 准备评测集</div><div class="section-title-row"><div><h2>选择或上传数据</h2><p class="muted">支持 JSONL / CSV，每行一个问题。</p></div><div class="template-actions"><button class="btn small" @click="downloadTemplate('jsonl')">↓ JSONL 模板</button><button class="btn small" @click="downloadTemplate('csv')">↓ CSV 模板</button></div></div><div class="dataset-tabs"><button :class="{ active: evalSource === 'golden' }" @click="evalSource = 'golden'">内置黄金集 <span>15</span></button><button :class="{ active: evalSource === 'upload' }" @click="evalSource = 'upload'">我的评测集 <span v-if="hasDataset">{{ datasetCases.length }}</span></button><button :class="{ active: evalSource === 'custom' }" @click="evalSource = 'custom'">快速粘贴</button></div><div v-if="evalSource === 'upload'" class="upload-zone" :class="{ dragover }" @dragover.prevent="dragover = true" @dragleave.prevent="dragover = false" @drop.prevent="onDrop" @click="uploadInput?.click()"><input ref="uploadInput" type="file" accept=".jsonl,.ndjson,.csv" hidden @change="handleFile($event.target.files[0])" /><div class="upload-icon">↥</div><strong>{{ datasetFile ? datasetFile.name : '拖拽测试集到这里' }}</strong><span>{{ datasetFile ? `${datasetCases.length} 条样本 · 已通过基础校验` : '或点击选择文件' }}</span><small>单文件最大 10 MB</small></div><div v-if="evalSource === 'custom'" class="custom-box"><p class="muted lab-desc">每行一题，可用 <code>=></code> 添加期望关键词，多个关键词用 <code>|</code> 分隔。</p><textarea v-model="customCasesText" class="textarea code" rows="5" placeholder="问题 => 期望关键词"></textarea></div><div v-if="evalSource === 'golden'" class="golden-summary"><span class="golden-icon">✦</span><div><strong>RAG 基础黄金集</strong><p>覆盖正向、陷阱、信息不足、全库检索四类场景，适合快速回归。</p></div><span class="sample-count">15 题</span></div><div v-if="datasetError" class="status err">{{ datasetError }}</div></div>
      <div class="card eval-config-card"><div class="section-kicker">02 / 配置运行</div><h2>选择评测策略</h2><p class="muted">先用 fake 模式快速检查，再用 real 模式验证真实模型表现。</p><div class="mode-switch"><button :class="{ active: evalMode === 'fake' }" @click="evalMode = 'fake'"><span>⚡</span><div><strong>Fake 冒烟</strong><small>快速 · 无模型成本</small></div></button><button :class="{ active: evalMode === 'real' }" @click="evalMode = 'real'"><span>◈</span><div><strong>Real 真实模型</strong><small>完整链路 · 约 20–30 秒</small></div></button></div><div class="metric-pick-title"><span>关注指标</span><span class="muted">{{ selectedMetrics.length }} 已选择</span></div><div class="metric-picks"><button v-for="metric in metricOptions" :key="metric.key" class="metric-pick" :class="{ selected: selectedMetrics.includes(metric.key) }" @click="toggleMetric(metric.key)"><span class="check">{{ selectedMetrics.includes(metric.key) ? '✓' : '' }}</span><span><strong>{{ metric.label }}</strong><small>{{ metric.hint }}</small></span></button></div><button class="btn primary run-btn" :disabled="evalRunning || (evalSource === 'upload' && !hasDataset)" @click="startEval"><span>{{ evalRunning ? '◌' : '▶' }}</span>{{ evalRunning ? '评测运行中…' : '开始评测' }}</button></div></section>

    <section v-if="evalRunning" class="card run-progress"><div class="progress-head"><div><strong>正在运行评测</strong><p class="muted">正在调用问答链路并记录逐题结果，请稍候…</p></div><span class="spinner">◌</span></div><div class="progress-track"><span></span></div><div class="progress-steps"><span class="done">✓ 读取数据</span><span class="active">◌ 执行问答</span><span>○ 计算指标</span><span>○ 生成报告</span></div></section>
    <section v-if="evalReport?.error" class="status err">{{ evalReport.error }}</section>
    <section v-if="evalReport && !evalReport.error" class="card report-card"><div class="report-header"><div><div class="section-kicker">03 / 评测结果</div><h2>本次运行表现</h2><p class="muted">{{ evalReport.results?.length || 0 }} 条样本 · {{ evalMode === 'fake' ? 'Fake 冒烟' : 'Real 真实模型' }} · 刚刚完成</p></div><button class="btn" @click="exportReport">↓ 导出 JSON 报告</button></div><div class="metric-row result-metrics"><div class="metric-card score-card"><span class="metric-value">{{ metricOf(evalReport.summary, 'pass_rate') }}</span><span class="metric-label">综合通过率</span><div class="score-ring">●</div></div><div v-for="metric in metricOptions.filter((item) => selectedMetrics.includes(item.key))" :key="metric.key" class="metric-card"><span class="metric-value">{{ metricOf(evalReport.summary, metric.key) }}</span><span class="metric-label">{{ metric.label }}</span></div><div class="metric-card"><span class="metric-value">{{ evalReport.summary?.avg_latency_ms ?? '—' }}<small v-if="evalReport.summary?.avg_latency_ms">ms</small></span><span class="metric-label">平均延迟</span></div></div><div class="result-toolbar"><div><strong>逐题明细</strong><span class="muted"> · 点击一行查看回答与引用</span></div><span class="result-legend"><span class="legend-dot pass"></span>通过 <span class="legend-dot fail"></span>需复盘</span></div><div class="result-list"><div v-for="result in evalReport.results" :key="result.id" class="result-row" :class="{ expanded: expandedResult === result.id }" @click="expandedResult = expandedResult === result.id ? null : result.id"><div class="result-index">{{ String(result.id).padStart(2, '0') }}</div><div class="result-question"><strong>{{ result.question }}</strong><small>{{ result.document || '全库检索' }}</small></div><span class="badge" :class="result.pass ? 'ok' : 'err'">{{ result.pass ? 'PASS' : 'FAIL' }}</span><div class="result-stat"><strong>{{ result.citations ?? 0 }}</strong><small>引用</small></div><div class="result-stat"><strong>{{ result.latency_ms ?? '—' }}<small v-if="result.latency_ms">ms</small></strong><small>耗时</small></div><span class="row-chevron">{{ expandedResult === result.id ? '⌃' : '⌄' }}</span><div v-if="expandedResult === result.id" class="result-detail"><p><b>评测问题</b> {{ result.question }}</p><p><b>结果状态</b> {{ result.pass ? '命中期望，回答通过' : '未满足期望，建议复盘回答或检索证据' }}</p></div></div></div><p v-if="evalReport.report_path" class="muted report-path">报告已保存：{{ evalReport.report_path }}</p></section>
    <section class="card history-card"><div class="section-kicker">运行记录</div><div class="history-row"><div class="history-icon">✓</div><div><strong>黄金集 · Fake 冒烟</strong><span>今天 14:32 · 15 题 · 综合通过率 86%</span></div><span class="tag-chip">已完成</span><button class="btn small">查看</button></div><div class="history-row muted-row"><div class="history-icon">↻</div><div><strong>上一次 Real 评测</strong><span>昨天 18:06 · 15 题 · 综合通过率 79%</span></div><span class="tag-chip">历史</span><button class="btn small">查看</button></div></section>

    <section class="card agent-card"><div class="lab-head"><div><div class="section-kicker">实验室 · 多 Agent</div><h2 class="lab-head-title">协作问答演练</h2><p class="muted lab-desc">Supervisor 规划 → 检索 Worker 召回 → 总结 Worker 作答，观察每一步的可解释输出。</p></div><span class="tag-chip">结构化可审计</span></div><div class="lab-controls wrap"><button v-for="question in presetQuestions" :key="question" class="chip" :class="{ active: agentQuestion === question }" @click="agentQuestion = question">{{ question }}</button><input v-model="agentQuestion" class="input grow" placeholder="自由输入问题…" @keydown.enter="runAgent" /><button class="btn primary" :disabled="agentRunning || !agentQuestion.trim()" @click="runAgent">{{ agentRunning ? '协作中…' : '运行' }}</button></div><div v-if="agentResult?.error" class="status err">{{ agentResult.error }}</div><div v-if="agentResult && !agentResult.error" class="agent-flow"><div class="agent-step plan"><div class="agent-head">🧠 Supervisor · 规划</div><p>查「<b>{{ agentResult.plan.doc_scope }}</b>」，检索词「{{ agentResult.plan.search_query }}」</p><p class="muted">{{ agentResult.plan.note }}</p></div><div class="agent-step retrieve"><div class="agent-head">🔍 检索 Worker · 命中 {{ agentResult.chunks.length }} 条</div><div v-for="(chunk, index) in agentResult.chunks.slice(0, 3)" :key="index" class="chunk-line"><span class="chunk-score">{{ (chunk.score * 100).toFixed(0) }}%</span><span class="chunk-text">{{ chunk.text.slice(0, 100) }}…</span></div></div><div class="agent-step answer"><div class="agent-head">✍️ 总结 Worker · 回答</div><p class="agent-answer">{{ agentResult.answer }}</p></div></div></section>
  </div>
</template>
