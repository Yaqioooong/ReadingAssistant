<script setup>
import { ref, computed, onMounted, onActivated } from 'vue'
import { getCacheStats, formatTime } from '../api.js'

const DAY_OPTIONS = [1, 7, 30]
const CHANNEL_LABELS = {
  exact: '精确命中',
  semantic: '语义命中',
  identifier: '标识符变体',
}
const CHANNEL_KEYS = ['exact', 'semantic', 'identifier']

const days = ref(7)
const data = ref(null)
const loading = ref(false)
const error = ref('')
let loadedOnce = false

async function load() {
  loading.value = true
  error.value = ''
  try {
    data.value = await getCacheStats(days.value)
    loadedOnce = true
  } catch (err) {
    error.value = err.message || '加载失败'
    data.value = null
  } finally {
    loading.value = false
  }
}

function switchDays(value) {
  if (days.value === value) return
  days.value = value
  load()
}

onMounted(() => {
  if (!loadedOnce) load()
})
onActivated(() => {
  if (!loadedOnce) load()
})

// ---------- 派生指标 ----------
const summary = computed(() => data.value?.summary || {})
const byChannel = computed(() => data.value?.by_channel || {})
const semantic = computed(() => data.value?.semantic || {})
const invalidated = computed(() => data.value?.invalidated || {})
const feedback = computed(() => data.value?.feedback || {})
const entries = computed(() => data.value?.entries || {})
const daily = computed(() => data.value?.daily || [])

/** 命中通道总量（不含 miss / disabled），作为通道分布条形图的分母 */
const channelTotal = computed(() =>
  CHANNEL_KEYS.reduce((sum, key) => sum + (byChannel.value[key] || 0), 0),
)

const channelRows = computed(() =>
  CHANNEL_KEYS.map((key) => {
    const count = byChannel.value[key] || 0
    return {
      key,
      label: CHANNEL_LABELS[key],
      count,
      share: channelTotal.value ? count / channelTotal.value : 0,
    }
  }),
)

/** 命中率：优先用可缓存口径（排除闲聊/多文档），旧后端无该字段时回退全量 */
const hitRate = computed(() => {
  const s = summary.value
  return s.hit_rate_cacheable != null ? s.hit_rate_cacheable : s.hit_rate
})

/** 单次命中相对未命中节省的耗时；数据不足时返回 null */
const savedPerHit = computed(() => {
  const hit = summary.value.avg_latency_hit_ms
  const miss = summary.value.avg_latency_miss_ms
  if (hit == null || miss == null) return null
  return miss - hit
})

const dailyMax = computed(() =>
  daily.value.reduce((max, day) => Math.max(max, day.hit || 0, day.miss || 0), 0),
)

/** 相似度刻度条：把 p50 与阈值映射到 0~1 的相对位置（阈值±0.05 的放大视图） */
const scoreScale = computed(() => {
  const threshold = semantic.value.threshold ?? 0
  const p50 = semantic.value.p50
  const span = 0.1
  const lo = Math.max(threshold - span / 2, 0)
  const toPos = (v) => {
    if (v == null) return null
    return Math.min(Math.max((v - lo) / span, 0), 1) * 100
  }
  return {
    thresholdPos: toPos(threshold),
    p50Pos: toPos(p50),
    p50,
    threshold,
  }
})

// ---------- 格式化 ----------
function pct(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}
function score(value) {
  return value == null ? '—' : Number(value).toFixed(3)
}
function ms(value) {
  return value == null ? '—' : `${Math.round(value)}ms`
}
function num(value) {
  return value == null ? 0 : value
}
</script>

<template>
  <section class="stats-panel">
    <header class="stats-head">
      <div>
        <h2 class="stats-title">缓存指标</h2>
        <p class="stats-sub">
          三层缓存（精确 / 语义 / 标识符变体）的命中归因、相似度分布与误命中情况
        </p>
      </div>
      <div class="stats-actions">
        <div class="day-group">
          <button
            v-for="option in DAY_OPTIONS"
            :key="option"
            class="day-btn"
            :class="{ active: days === option }"
            @click="switchDays(option)"
          >
            {{ option }} 天
          </button>
        </div>
        <button class="refresh-btn" :disabled="loading" @click="load">
          {{ loading ? '加载中…' : '刷新' }}
        </button>
      </div>
    </header>

    <p v-if="error" class="stats-error">
      {{ error }}
      <button class="retry-btn" @click="load">重试</button>
    </p>

    <div v-if="data" class="stats-body">
      <!-- KPI -->
      <div class="kpi-row">
        <div class="card kpi">
          <span class="kpi-label">缓存命中率</span>
          <span class="kpi-value">{{ pct(hitRate) }}</span>
          <span v-if="summary.cacheable_requests != null" class="kpi-foot">
            {{ num(summary.hit_total) }} 命中 / {{ num(summary.cacheable_requests) }} 可缓存请求
            <template v-if="summary.total_requests !== summary.cacheable_requests">
              （已排除 {{ summary.total_requests - summary.cacheable_requests }} 轮闲聊/多文档）
            </template>
          </span>
          <span v-else class="kpi-foot">
            {{ num(summary.hit_total) }} 命中 / {{ num(summary.total_requests) }} 请求
          </span>
        </div>
        <div class="card kpi">
          <span class="kpi-label">命中总量</span>
          <span class="kpi-value">{{ num(summary.hit_total) }}</span>
          <span class="kpi-foot">节省 {{ num(summary.saved_calls) }} 次模型调用</span>
        </div>
        <div class="card kpi">
          <span class="kpi-label">平均节省耗时</span>
          <span class="kpi-value">{{ ms(savedPerHit) }}</span>
          <span class="kpi-foot">
            命中 {{ ms(summary.avg_latency_hit_ms) }} / 未命中 {{ ms(summary.avg_latency_miss_ms) }}
          </span>
        </div>
        <div class="card kpi" :class="{ warn: feedback.mis_hit_rate > 0 }">
          <span class="kpi-label">误命中率</span>
          <span class="kpi-value">{{ pct(feedback.mis_hit_rate) }}</span>
          <span class="kpi-foot">
            <template v-if="feedback.cache_hit_feedback">
              缓存答案被点踩 {{ num(feedback.cache_hit_down) }} /
              反馈 {{ num(feedback.cache_hit_feedback) }}
            </template>
            <template v-else>暂无反馈数据</template>
          </span>
        </div>
      </div>

      <div class="stats-grid">
        <!-- 通道分布 -->
        <div class="card stats-card">
          <div class="card-head">
            <h3>命中通道分布</h3>
            <span class="muted">共 {{ channelTotal }} 次命中</span>
          </div>
          <div v-if="channelTotal" class="channel-list">
            <div v-for="row in channelRows" :key="row.key" class="channel-row">
              <span class="channel-name">{{ row.label }}</span>
              <div class="channel-track">
                <div class="channel-fill" :style="{ width: `${row.share * 100}%` }"></div>
              </div>
              <span class="channel-count">
                {{ row.count }}<span class="muted">
                  （{{ pct(row.share) }}）</span>
              </span>
            </div>
          </div>
          <p v-else class="muted empty-hint">窗口内暂无缓存命中</p>
          <div class="channel-foot">
            <span>未命中 {{ num(byChannel.miss) }}</span>
            <span>
              跳过缓存 {{ num(byChannel.disabled) + num(byChannel.skipped) }}
              （多文档 {{ num(byChannel.disabled) }} / 闲聊·历史 {{ num(byChannel.skipped) }}）
            </span>
          </div>
        </div>

        <!-- 语义相似度 -->
        <div class="card stats-card">
          <div class="card-head">
            <h3>语义命中相似度</h3>
            <span class="muted">样本 {{ num(semantic.count) }}</span>
          </div>
          <div class="score-scale">
            <div class="score-track">
              <div class="score-threshold" :style="{ left: `${scoreScale.thresholdPos}%` }"></div>
              <div
                v-if="scoreScale.p50Pos != null"
                class="score-p50"
                :style="{ left: `${scoreScale.p50Pos}%` }"
              ></div>
            </div>
            <div class="score-legend">
              <span>阈值 {{ score(scoreScale.threshold) }}</span>
              <span v-if="scoreScale.p50 != null">P50 {{ score(scoreScale.p50) }}</span>
            </div>
          </div>
          <div class="score-grid">
            <div><span class="muted">P50</span><b>{{ score(semantic.p50) }}</b></div>
            <div><span class="muted">P95</span><b>{{ score(semantic.p95) }}</b></div>
            <div><span class="muted">最小</span><b>{{ score(semantic.min) }}</b></div>
            <div><span class="muted">最大</span><b>{{ score(semantic.max) }}</b></div>
          </div>
          <p class="score-note">
            擦肩而过（距阈值 &lt;0.01）<b>{{ num(semantic.near_threshold) }}</b> 次
            ——下调阈值可多拿的命中量级
          </p>
          <p class="score-note">
            脏缓存丢弃 <b>{{ num(invalidated.count) }}</b> 次（{{ pct(invalidated.rate) }}），
            每次都会重跑完整检索
          </p>
        </div>
      </div>

      <div class="stats-grid">
        <!-- 每日趋势 -->
        <div class="card stats-card">
          <div class="card-head">
            <h3>每日命中趋势</h3>
            <span class="legend">
              <i class="dot hit"></i>命中<i class="dot miss"></i>未命中
            </span>
          </div>
          <div v-if="dailyMax" class="trend">
            <div v-for="day in daily" :key="day.date" class="trend-col">
              <div class="trend-bars">
                <div
                  class="trend-bar hit"
                  :style="{ height: `${(day.hit / dailyMax) * 100}%` }"
                  :title="`${day.date} 命中 ${day.hit}`"
                ></div>
                <div
                  class="trend-bar miss"
                  :style="{ height: `${(day.miss / dailyMax) * 100}%` }"
                  :title="`${day.date} 未命中 ${day.miss}`"
                ></div>
              </div>
              <span class="trend-date">{{ day.date.slice(5) }}</span>
            </div>
          </div>
          <p v-else class="muted empty-hint">窗口内暂无请求</p>
        </div>

        <!-- 存量 -->
        <div class="card stats-card">
          <div class="card-head">
            <h3>缓存存量</h3>
            <span class="muted">累计命中 {{ num(entries.total_hits) }} 次</span>
          </div>
          <div class="entry-grid">
            <div><span class="muted">总条目</span><b>{{ num(entries.total) }}</b></div>
            <div>
              <span class="muted">冷条目</span><b>{{ num(entries.cold) }}</b>
            </div>
            <div><span class="muted">HITL 式</span><b>{{ num(entries.hitl) }}</b></div>
            <div>
              <span class="muted">带引用</span><b>{{ num(entries.with_citations) }}</b>
            </div>
            <div>
              <span class="muted">临近过期</span><b>{{ num(entries.expiring_soon) }}</b>
            </div>
            <div>
              <span class="muted">非缓存错误率</span>
              <b>{{ pct(feedback.non_cache_error_rate) }}</b>
            </div>
          </div>
        </div>
      </div>

      <!-- 热门缓存 -->
      <div class="card stats-card">
        <div class="card-head">
          <h3>热门缓存 Top 10</h3>
          <span class="muted">按命中次数排序</span>
        </div>
        <table v-if="entries.top_hits && entries.top_hits.length" class="top-table">
          <thead>
            <tr>
              <th class="col-idx">#</th>
              <th>问题</th>
              <th class="col-num">命中</th>
              <th class="col-time">最近命中</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(item, index) in entries.top_hits" :key="index">
              <td class="col-idx muted">{{ index + 1 }}</td>
              <td class="col-q" :title="item.question">{{ item.question }}</td>
              <td class="col-num"><b>{{ item.hit_count }}</b></td>
              <td class="col-time muted">{{ formatTime(item.last_hit_at) }}</td>
            </tr>
          </tbody>
        </table>
        <p v-else class="muted empty-hint">窗口内暂无命中记录</p>
      </div>
    </div>

    <p v-else-if="loading" class="muted empty-hint">加载中…</p>
  </section>
</template>
