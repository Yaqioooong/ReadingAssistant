<script setup>
import { ref, onMounted } from 'vue'
import { getSettings, updateSettings } from '../api.js'

const groups = ref([])
const loading = ref(true)
const savingId = ref(null)
const toast = ref(null)
const secretInputs = ref({}) // key -> 新输入值
const clearSecrets = ref({}) // key -> 是否清除
let toastTimer = null

function showToast(message, type = 'ok') {
  toast.value = { message, type }
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => (toast.value = null), 3200)
}

async function load() {
  loading.value = true
  try {
    const data = await getSettings()
    groups.value = data.groups || []
    secretInputs.value = {}
    clearSecrets.value = {}
  } catch (err) {
    showToast(`读取配置失败：${err.message}`, 'err')
  } finally {
    loading.value = false
  }
}

function itemTouched(item) {
  return item && !item._touched ? ((item._touched = true), true) : true
}

function valueOf(item) {
  const v = item.value
  if (item.type === 'bool') return v === true || v === 'true'
  return v
}

async function saveGroup(group) {
  if (savingId.value) return
  const updates = {}
  for (const item of group.items) {
    if (item.type === 'secret') {
      const typed = (secretInputs.value[item.key] || '').trim()
      if (clearSecrets.value[item.key]) updates[item.key] = '__CLEAR__'
      else if (typed) updates[item.key] = typed
      continue
    }
    if (item.type === 'readonly') continue
    updates[item.key] = String(item.value)
  }
  if (!Object.keys(updates).length) {
    showToast('没有需要保存的改动', 'ok')
    return
  }
  savingId.value = group.id
  try {
    const resp = await updateSettings(updates)
    const parts = []
    if (resp.applied?.length) parts.push(`更新 ${resp.applied.length} 项`)
    if (resp.cleared?.length) parts.push(`清除 ${resp.cleared.length} 项`)
    showToast(`${parts.join('，') || '已保存'} · 配置即时生效`)
    await load()
  } catch (err) {
    showToast(`保存失败：${err.message}`, 'err')
  } finally {
    savingId.value = null
  }
}
onMounted(load)
</script>

<template>
  <div class="settings-page">
    <header class="settings-head">
      <div>
        <h2 class="settings-title">⚙️ 设置</h2>
        <p class="muted settings-sub">配置保存在本地 .env，修改后即时生效，无需重启</p>
      </div>
      <span class="settings-badge">
        <i class="dot-live"></i>本地持久化
      </span>
    </header>

    <div v-if="loading" class="settings-loading muted">加载配置中…</div>

    <div v-for="group in groups" :key="group.id"
      class="card settings-card"
      :class="{ wide: group.id === 'advanced' }">
      <div class="sc-head">
        <div class="sc-title-wrap">
          <h3 class="sc-title">{{ group.title }}</h3>
          <p v-if="group.desc" class="muted sc-desc">{{ group.desc }}</p>
        </div>
        <button
          v-if="group.id !== 'advanced'"
          class="btn primary sc-save"
          :disabled="savingId === group.id"
          @click="saveGroup(group)"
        >
          {{ savingId === group.id ? '保存中…' : '保存' }}
        </button>
      </div>

      <div class="sc-body">
        <template v-for="item in group.items" :key="item.key">
          <!-- 只读 -->
          <div v-if="item.type === 'readonly'" class="set-row readonly">
            <div class="set-info">
              <span class="set-label">{{ item.label }}</span>
              <span class="set-hint muted">只读 · 需源码或重启</span>
            </div>
            <code class="set-readonly-value">{{ item.value || '—' }}</code>
          </div>

          <!-- 布尔开关 -->
          <div v-else-if="item.type === 'bool'" class="set-row">
            <div class="set-info">
              <span class="set-label">{{ item.label }}</span>
              <span v-if="item.hint" class="set-hint muted">{{ item.hint }}</span>
            </div>
            <label class="sw" :class="{ on: valueOf(item) }">
              <input v-model="item.value" type="checkbox" @change="itemTouched(item)" />
              <span class="sw-track"></span>
            </label>
          </div>

          <!-- 密钥（脱敏展示） -->
          <div v-else-if="item.type === 'secret'" class="set-row secret">
            <div class="set-info">
              <span class="set-label">{{ item.label }}</span>
              <span
                class="secret-state"
                :class="{ configured: item.state?.configured, missing: !item.state?.configured }"
              >
                <i class="dot-live"></i>
                {{ item.state?.configured ? `已配置 · ${item.state.masked}` : '未配置' }}
              </span>
            </div>
            <div class="secret-controls">
              <input
                v-model="secretInputs[item.key]"
                class="input secret-input"
                type="password"
                autocomplete="off"
                :placeholder="item.state?.configured ? '留空保持不变，输入新密钥可覆盖' : '粘贴 API Key…'"
              />
              <button
                v-if="item.state?.configured"
                type="button"
                class="btn danger-ghost"
                :disabled="savingId === group.id"
                @click="clearSecrets[item.key] = !clearSecrets[item.key]"
              >
                {{ clearSecrets[item.key] ? '待清除 ✓' : '清除' }}
              </button>
            </div>
          </div>

          <!-- 数值 -->
          <div v-else-if="['int', 'float'].includes(item.type)" class="set-row">
            <div class="set-info">
              <span class="set-label">{{ item.label }}</span>
              <span v-if="item.hint" class="set-hint muted">{{ item.hint }}</span>
            </div>
            <input
              v-model="item.value"
              class="input num"
              type="number"
              :step="item.step || (item.type === 'float' ? 0.01 : 1)"
              :min="item.min"
              :max="item.max"
              @input="itemTouched(item)"
            />
          </div>

          <!-- 单选 -->
          <div v-else-if="item.type === 'choice'" class="set-row">
            <div class="set-info">
              <span class="set-label">{{ item.label }}</span>
              <span v-if="item.hint" class="set-hint muted">{{ item.hint }}</span>
            </div>
            <div class="choice-group">
              <label
                v-for="[value, label] in item.choices"
                :key="value"
                class="choice-pill"
                :class="{ active: item.value === value }"
              >
                <input v-model="item.value" type="radio" :value="value" @change="itemTouched(item)" />
                <span class="pill-dot"></span>
                <span>{{ label }}</span>
              </label>
            </div>
          </div>

          <!-- 文本 -->
          <div v-else class="set-row">
            <div class="set-info">
              <span class="set-label">{{ item.label }}</span>
              <span v-if="item.hint" class="set-hint muted">{{ item.hint }}</span>
            </div>
            <input v-model="item.value" class="input" type="text" @input="itemTouched(item)" />
          </div>
        </template>
      </div>
    </div>

    <transition name="ra-toast">
      <div v-if="toast" class="ra-toast" :class="toast.type">{{ toast.message }}</div>
    </transition>
  </div>
</template>

<style>
.settings-page {
  width: 100%;
  margin: 0 auto;
  padding: 0 0 60px;
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  align-items: start;
}

/* 桌面宽屏：主内容 2 列，不超出外层容器 */
@media (min-width: 1180px) {
  .settings-page { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .settings-page .settings-card.wide { grid-column: 1 / -1; }
}
.settings-head { grid-column: 1 / -1; display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 6px 2px 2px; }
.settings-title { margin: 0; font-size: 20px; }
.settings-sub { margin: 4px 0 0; font-size: 13px; }
.settings-badge { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 6px 12px; }
.settings-badge .dot-live, .secret-state .dot-live { width: 7px; height: 7px; border-radius: 50%; background: var(--ok); display: inline-block; }
.settings-loading { grid-column: 1 / -1; padding: 40px 0; text-align: center; }
.settings-card { padding: 18px 20px; display: flex; flex-direction: column; gap: 6px; }
.settings-card.wide { grid-column: 1 / -1; }
.sc-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }
.sc-title { margin: 0; font-size: 15.5px; }
.sc-desc { margin: 4px 0 0; font-size: 12.5px; line-height: 1.6; }
.sc-save { flex-shrink: 0; }
.sc-body { display: flex; flex-direction: column; margin-top: 6px; }
.set-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 11px 2px; border-top: 1px solid var(--border); }
.set-row:first-child { border-top: none; }
.set-info { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.set-label { font-size: 13.5px; font-weight: 500; }
.set-hint { font-size: 11.5px; line-height: 1.5; }
.set-row .input { width: min(46%, 260px); }
.set-row .input.num { width: 150px; }
.secret-input { width: min(52%, 300px) !important; }
.set-readonly-value { font-size: 12px; color: var(--muted); background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 5px 9px; max-width: 340px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.secret-state { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; }
.secret-state.configured { color: var(--ok); }
.secret-state.missing .dot-live { background: var(--muted); }
.secret-state.missing { color: var(--muted); }
.secret-controls { display: flex; gap: 8px; }
.secret-input { width: 240px !important; font-family: ui-monospace, Menlo, monospace; }
.btn.danger-ghost { background: transparent; border: 1px solid var(--err); color: var(--err); border-radius: 10px; padding: 6px 12px; font-size: 12.5px; cursor: pointer; }
.btn.danger-ghost:hover { background: color-mix(in srgb, var(--err) 12%, transparent); }

/* 开关 */
.sw { display: inline-flex; cursor: pointer; }
.sw input { display: none; }
.sw-track { width: 40px; height: 22px; border-radius: 999px; background: var(--border); position: relative; transition: background-color 0.2s; }
.sw-track::after { content: ''; position: absolute; top: 2.5px; left: 3px; width: 17px; height: 17px; border-radius: 50%; background: var(--card); box-shadow: 0 1px 3px rgba(0,0,0,.3); transition: transform 0.18s ease; }
.sw.on .sw-track { background: var(--accent); }
.sw.on .sw-track::after { transform: translateX(17px); }

/* 单选胶囊 */
.choice-group { display: flex; gap: 8px; flex-wrap: wrap; }
.choice-pill { display: inline-flex; align-items: center; gap: 7px; padding: 7px 13px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg); font-size: 12.5px; color: var(--muted); cursor: pointer; transition: all 0.15s ease; }
.choice-pill input { display: none; }
.pill-dot { width: 7px; height: 7px; border-radius: 50%; background: transparent; border: 1.5px solid var(--border); transition: all 0.15s ease; }
.choice-pill.active { border-color: var(--accent); color: var(--text); background: var(--accent-weak); }
.choice-pill.active .pill-dot { background: var(--accent); border-color: var(--accent); }

/* Toast */
.ra-toast { position: fixed; bottom: 28px; left: 50%; transform: translateX(-50%); z-index: 999; padding: 10px 18px; border-radius: 12px; font-size: 13px; background: var(--card); border: 1px solid var(--ok); color: var(--text); box-shadow: 0 10px 30px rgba(0,0,0,.25); }
.ra-toast.err { border-color: var(--err); }
.ra-toast-enter-active, .ra-toast-leave-active { transition: opacity 0.2s ease, transform 0.2s ease; }
.ra-toast-enter-from, .ra-toast-leave-to { opacity: 0; transform: translateX(-50%) translateY(8px); }

@media (max-width: 720px) {
  .set-row { flex-direction: column; align-items: flex-start; gap: 10px; }
  .set-row .input, .secret-input { width: 100% !important; }
}
</style>
