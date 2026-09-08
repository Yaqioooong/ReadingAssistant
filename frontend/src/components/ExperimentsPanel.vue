<script setup>
import { ref } from 'vue'
import { runEval, runMultiAgent } from '../api.js'

// ---------- 评测中心 ----------
const evalMode = ref('fake')
const evalSource = ref('golden')
const evalRunning = ref(false)
const evalReport = ref(null)
const customCasesText = ref(
  '张三喜欢谁？ => 李四\n王五喜欢李四吗？ => 不|没|否\n少白公的原名叫什么？'
)

const evalMetrics = [
  { key: 'pass_rate', label: '整体通过率' },
  { key: 'answerable_accuracy', label: '可答准确率' },
  { key: 'unanswerable_recognition', label: '不可答识别率' },
  { key: 'citation_coverage', label: '引用覆盖率' },
]

function metricOf(summary, key) {
  const v = summary?.[key]
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(0)}%`
}

function parseCustomCases(text) {
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [q, kw] = line.split('=>').map((s) => s.trim())
      return {
        question: q,
        expect_keywords: kw ? kw.split('|').map((s) => s.trim()).filter(Boolean) : [],
      }
    })
}

async function startEval() {
  if (evalRunning.value) return
  const custom =
    evalSource.value === 'custom' ? parseCustomCases(customCasesText.value) : []
  if (evalSource.value === 'custom' && !custom.length) return
  evalRunning.value = true
  evalReport.value = null
  try {
    evalReport.value = await runEval(evalMode.value, 0, custom)
  } catch (err) {
    evalReport.value = { error: err.message }
  } finally {
    evalRunning.value = false
  }
}

// ---------- 多 Agent 工作台 ----------
const presetQuestions = ['张三喜欢谁？', '罗辑在第三章放下了什么？', '少白公的结局是什么？']
const agentQuestion = ref(presetQuestions[0])
const agentRunning = ref(false)
const agentResult = ref(null)

async function runAgent() {
  const text = agentQuestion.value.trim()
  if (!text || agentRunning.value) return
  agentRunning.value = true
  agentResult.value = null
  try {
    agentResult.value = await runMultiAgent(text)
  } catch (err) {
    agentResult.value = { error: err.message }
  } finally {
    agentRunning.value = false
  }
}
</script>

<template>
  <div class="lab">
    <!-- ============ 0. 页首导语 ============ -->
    <section class="lab-hero">
      <div class="lab-hero-badges">
        <span class="tag-chip experimental">实验性 · Experimental</span>
        <span class="tag-chip">与生产链路同源</span>
      </div>
      <h1 class="lab-hero-title">🧪 实验室</h1>
      <p class="lab-hero-desc">
        面向工程验证的工作台，两块能力：<b>量化评测 RAG 质量</b> 与 <b>演练多 Agent 协作范式</b>。
        所有实验复用生产的解析 / 入库 / 检索 / 问答链路，输出结构化报告，可作为迭代与验收的量化依据。
      </p>
      <div class="lab-hero-grid">
        <div class="lab-hero-point">
          <span class="ph-ico">🎯</span>
          <div>
            <b>评测方法论</b>
            <p>黄金集四题型（正向 / 陷阱 / 信息不足 / 全库）× fake 冒烟与 real 真实链路双模式 × 四指标量化（准确率 · 不可答识别 · 引用覆盖 · 延迟）</p>
          </div>
        </div>
        <div class="lab-hero-point">
          <span class="ph-ico">🤝</span>
          <div>
            <b>多 Agent 协作范式</b>
            <p>Supervisor 通过函数调用规划检索范围 → 检索 Worker 召回 → 总结 Worker 带引用作答；共享生产问答图的检索与引用链路</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ============ 1. 评测中心 ============ -->
    <section class="card">
      <div class="lab-head">
        <div>
          <h2 class="lab-head-title">RAG 评测中心</h2>
          <p class="muted lab-desc">
            黄金集 15 题（正向 / 陷阱 / 信息不足 / 全库四类）或自定义题目；一键跑通端到端 API 链路，
            产出质量指标与逐题明细，报告自动落盘 <code>eval/reports/</code>。
          </p>
        </div>
        <div class="lab-head-tags">
          <span class="tag-chip">15 题黄金集</span>
          <span class="tag-chip">fake 冒烟</span>
          <span class="tag-chip">real 真实链路</span>
          <span class="tag-chip">四指标量化</span>
        </div>
      </div>

      <div class="lab-controls">
        <span class="ctl-label">模型：</span>
        <label class="radio-inline" :class="{ active: evalMode === 'fake' }">
          <input v-model="evalMode" type="radio" value="fake" /> fake 冒烟（快）
        </label>
        <label class="radio-inline" :class="{ active: evalMode === 'real' }">
          <input v-model="evalMode" type="radio" value="real" /> real 真实模型
        </label>
        <span class="ctl-label sep">题目：</span>
        <label class="radio-inline" :class="{ active: evalSource === 'golden' }">
          <input v-model="evalSource" type="radio" value="golden" /> 黄金集 15 题
        </label>
        <label class="radio-inline" :class="{ active: evalSource === 'custom' }">
          <input v-model="evalSource" type="radio" value="custom" /> 自定义题目
        </label>
        <button class="btn primary" :disabled="evalRunning" @click="startEval">
          {{ evalRunning ? '评测运行中…' : '运行评测' }}
        </button>
      </div>

      <div v-if="evalSource === 'custom'" class="custom-box">
        <p class="muted lab-desc">
          每行一题；可加 <code>=> 期望关键词</code>（多个词用 <code>|</code> 分隔）。
          无关键词的题目只检查能否给出回答。默认全库检索。
        </p>
        <textarea
          v-model="customCasesText"
          class="textarea code"
          rows="4"
          placeholder="每行一题，如：张三喜欢谁？ => 李四"
        ></textarea>
      </div>

      <div v-if="evalReport?.error" class="status err">{{ evalReport.error }}</div>

      <template v-if="evalReport && !evalReport.error">
        <div class="metric-row">
          <div v-for="m in evalMetrics" :key="m.key" class="metric-card">
            <span class="metric-value">{{ metricOf(evalReport.summary, m.key) }}</span>
            <span class="metric-label">{{ m.label }}</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{{ evalReport.summary.avg_latency_ms }}ms</span>
            <span class="metric-label">平均延迟</span>
          </div>
        </div>
        <table class="table lab-table">
          <thead>
            <tr>
              <th>#</th><th>题目</th><th>范围</th><th>结果</th><th>引用</th><th>耗时</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in evalReport.results" :key="r.id">
              <td>{{ r.id }}</td>
              <td class="q-cell">{{ r.question }}</td>
              <td class="muted">{{ r.document || '(全库)' }}</td>
              <td>
                <span class="badge" :class="r.pass ? 'ok' : 'err'">
                  {{ r.pass ? 'PASS' : 'FAIL' }}
                </span>
              </td>
              <td>{{ r.citations }}</td>
              <td>{{ r.latency_ms }}ms</td>
            </tr>
          </tbody>
        </table>
        <p v-if="evalReport.report_path" class="muted lab-hint">
          报告已存：{{ evalReport.report_path }}
        </p>
      </template>
    </section>

    <!-- ============ 2. 多 Agent 工作台 ============ -->
    <section class="card">
      <div class="lab-head">
        <div>
          <h2 class="lab-head-title">多 Agent 协作问答</h2>
          <p class="muted lab-desc">
            演练 <b>Supervisor → Worker</b> 编排范式：Supervisor 用函数调用（make_plan）输出检索计划（范围 + 检索词 + 理由），
            检索 Worker 执行召回，总结 Worker 综合作答。每一步状态可观测，便于对照“流水线 vs 多 Agent”的取舍。
          </p>
        </div>
        <div class="lab-head-tags">
          <span class="tag-chip">函数调用规划</span>
          <span class="tag-chip">Supervisor / Worker</span>
          <span class="tag-chip">结构化计划可审计</span>
        </div>
      </div>
      <div class="lab-controls wrap">
        <button
          v-for="q in presetQuestions"
          :key="q"
          class="chip"
          :class="{ active: agentQuestion === q }"
          @click="agentQuestion = q"
        >
          {{ q }}
        </button>
        <input
          v-model="agentQuestion"
          class="input grow"
          placeholder="自由输入想读的问题…"
          @keydown.enter="runAgent"
        />
        <button class="btn primary" :disabled="agentRunning || !agentQuestion.trim()" @click="runAgent">
          {{ agentRunning ? '协作中…' : '运行' }}
        </button>
      </div>

      <div v-if="agentResult?.error" class="status err">{{ agentResult.error }}</div>

      <div v-if="agentResult && !agentResult.error" class="agent-flow">
        <div class="agent-step plan">
          <div class="agent-head">🧠 Supervisor · 规划</div>
          <p>
            查「<b>{{ agentResult.plan.doc_scope }}</b>」，检索词「{{ agentResult.plan.search_query }}」
          </p>
          <p class="muted">{{ agentResult.plan.note }}</p>
        </div>
        <div class="agent-step retrieve">
          <div class="agent-head">🔍 检索 Worker · 命中 {{ agentResult.chunks.length }} 条</div>
          <div v-for="(c, i) in agentResult.chunks.slice(0, 3)" :key="i" class="chunk-line">
            <span class="chunk-score">{{ (c.score * 100).toFixed(0) }}%</span>
            <span class="chunk-text">{{ c.text.slice(0, 80) }}…</span>
          </div>
          <p v-if="!agentResult.chunks.length" class="muted">未检索到相关片段</p>
        </div>
        <div class="agent-step answer">
          <div class="agent-head">✍️ 总结 Worker · 回答</div>
          <p class="agent-answer">{{ agentResult.answer }}</p>
          <div v-if="agentResult.citations?.length" class="citations">
            <div v-for="(c, j) in agentResult.citations" :key="j" class="citation">
              <span class="src">[{{ j + 1 }}] {{ c.chapter || '引用' }}</span>
              <div class="excerpt">{{ c.excerpt }}</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style>
.lab { display: flex; flex-direction: column; gap: 18px; max-width: 960px; margin: 0 auto; padding-bottom: 40px; }
.lab-desc { font-size: 13px; margin: 4px 0 12px; }
.lab-controls { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.lab-controls.wrap { gap: 8px; }
.ctl-label { font-size: 13px; color: var(--muted); }
.ctl-label.sep { margin-left: 8px; }
.radio-inline { display: flex; align-items: center; gap: 5px; font-size: 13px; cursor: pointer; padding: 6px 10px; border: 1px solid var(--border); border-radius: 8px; }
.radio-inline.active { border-color: var(--accent); color: var(--accent); }
.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 14px; }
.metric-card { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px; text-align: center; }
.metric-value { display: block; font-size: 20px; font-weight: 700; color: var(--accent); }
.metric-label { font-size: 12px; color: var(--muted); }
.lab-table td.q-cell { max-width: 280px; }
.custom-box { background: var(--bg); border: 1px dashed var(--border); border-radius: 10px; padding: 12px; margin-bottom: 14px; }
.textarea.code { width: 100%; font-family: ui-monospace, Menlo, monospace; font-size: 13px; background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 10px; resize: vertical; }
.chip { border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; padding: 6px 10px; border-radius: 999px; cursor: pointer; }
.chip.active { border-color: var(--accent); color: var(--accent); }
.input.grow { flex: 1; min-width: 200px; }
.agent-flow { display: flex; flex-direction: column; gap: 12px; }
.agent-step { border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 12px 14px; }
.agent-step.plan { border-left-color: #b57bff; }
.agent-step.retrieve { border-left-color: #3fb950; }
.agent-head { font-size: 13px; font-weight: 600; margin-bottom: 6px; }
.agent-answer { line-height: 1.7; }
.chunk-line { display: flex; gap: 8px; font-size: 12px; padding: 3px 0; }
.chunk-score { flex-shrink: 0; color: var(--ok); font-weight: 600; }
.chunk-text { color: var(--muted); }
.badge.ok { color: var(--ok); border: 1px solid var(--ok); border-radius: 999px; padding: 1px 8px; font-size: 12px; }
.badge.err { color: var(--err); border: 1px solid var(--err); border-radius: 999px; padding: 1px 8px; font-size: 12px; }
</style>

/* 实验台页 */
.lab { max-width: 1020px; }
.lab-hero {
  background: linear-gradient(135deg, var(--accent-weak), transparent 62%), var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 22px 24px;
}
.lab-hero-badges { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.tag-chip {
  display: inline-flex;
  align-items: center;
  font-size: 11.5px;
  line-height: 1;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
  background: var(--bg);
  white-space: nowrap;
}
.tag-chip.experimental { color: #b57bff; border-color: #b57bff55; background: rgba(181, 123, 255, 0.08); }
.lab-hero-title { margin: 0 0 6px; font-size: 22px; }
.lab-hero-desc { margin: 0 0 14px; font-size: 13.5px; line-height: 1.8; color: var(--text); opacity: 0.92; max-width: 820px; }
.lab-hero-desc b, .lab-desc b { color: var(--text); }
.lab-hero-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.lab-hero-point {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 14px;
  font-size: 12.5px;
}
.lab-hero-point p { margin: 4px 0 0; color: var(--muted); line-height: 1.7; }
.ph-ico { font-size: 18px; line-height: 1.3; }
.lab-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; }
.lab-head-title { margin: 0 0 6px; font-size: 17px; }
.lab-head-tags { display: flex; gap: 6px; flex-wrap: wrap; padding-top: 2px; }
