<script setup>
import { ref, computed, nextTick, watch, onMounted, onActivated } from 'vue'
import {
  createSession,
  listSessions,
  getHistory,
  sendMessage,
  deleteSession,
  listHitlTasks,
  submitClarification,
  rejectClarification,
  submitFeedback,
  formatTime,
} from '../api.js'

const props = defineProps({
  documents: { type: Array, default: () => [] },
})

const sessions = ref([])
const currentSessionId = ref(null)
const messages = ref([])
const question = ref('')
const selectedDocIds = ref([])  // 空数组 = 全部书籍
const scopeOpen = ref(false)
const scopeSearch = ref('')
const sending = ref(false)
// 待澄清任务：{ taskId, question } —— 信息不足时记录，用户补充后带澄清内容重发
const pendingClarification = ref(null)
const clarificationText = ref('')
const clarificationBusy = ref(false)

const filteredDocs = computed(() => {
  const query = scopeSearch.value.trim().toLowerCase()
  if (!query) return props.documents
  return props.documents.filter((doc) => doc.filename.toLowerCase().includes(query))
})

const scopeLabel = computed(() => {
  const n = selectedDocIds.value.length
  if (!n) return '全部书籍'
  if (n === 1) {
    const doc = props.documents.find((d) => d.id === selectedDocIds.value[0])
    return doc ? doc.filename : '已选 1 本'
  }
  return `已选 ${n} 本`
})
const messageBox = ref(null)
const stickToBottom = ref(true)

const currentTitle = computed(() => {
  const session = sessions.value.find((item) => item.id === currentSessionId.value)
  return session ? sessionLabel(session) : '新的对话'
})

function sessionLabel(session) {
  return `会话 ${session.id.slice(0, 8)}`
}

function toggleDoc(id) {
  const index = selectedDocIds.value.indexOf(id)
  if (index >= 0) {
    selectedDocIds.value.splice(index, 1)
  } else {
    selectedDocIds.value.push(id)
  }
}

// 书籍被删除/刷新后，清理已失效的多选 id
watch(
  () => props.documents.map((d) => d.id),
  (ids) => {
    selectedDocIds.value = selectedDocIds.value.filter((id) => ids.includes(id))
  },
)

async function loadSessions() {
  try {
    sessions.value = await listSessions()
  } catch {
    sessions.value = []
  }
}

async function createNewSession() {
  try {
    const { session_id } = await createSession()
    currentSessionId.value = session_id
    messages.value = []
    pendingClarification.value = null
    clarificationText.value = ''
    await loadSessions()
    scrollToBottom()
  } catch (err) {
    messages.value = [{ role: 'system', content: `出错：${err.message}` }]
  }
}

async function removeSession(id) {
  try {
    await deleteSession(id)
    sessions.value = sessions.value.filter((s) => s.id !== id)
    if (currentSessionId.value === id) {
      currentSessionId.value = null
      messages.value = []
    }
  } catch (err) {
    messages.value = [{ role: 'system', content: `删除失败：${err.message}` }]
  }
}

function mapHistory(history) {
  return history.map((m) => ({
    role: m.role === 'assistant' ? 'assistant' : m.role === 'user' ? 'user' : 'system',
    content: m.content,
    citations: m.meta?.citations || [],
    // 缓存来源：历史消息由 record 节点写入 meta，实时响应用 resp 字段
    cacheHit: !!(m.meta?.cache_hit),
    cacheChannel: m.meta?.cache_channel || null,
    feedback: null,
    feedbackBusy: false,
  }))
}

// 用服务端记录覆盖本地视图，保证「补充后重问」等场景刷新前后展示一致
async function reloadHistory() {
  if (!currentSessionId.value) return
  messages.value = mapHistory(await getHistory(currentSessionId.value))
  scrollToBottom()
}

async function selectSession(id) {
  currentSessionId.value = id
  try {
    await reloadHistory()
    await restorePendingClarification(id)
    scrollToBottom()
  } catch (err) {
    messages.value = [{ role: 'system', content: `出错：${err.message}` }]
  }
}

// 发起一次问答请求（不含用户气泡，由调用方负责）
async function runQuestion(text, clarification = null) {
  if (!currentSessionId.value) {
    const { session_id } = await createSession()
    currentSessionId.value = session_id
    await loadSessions()
  }
  const payload = { question: text, document_ids: [...selectedDocIds.value] }
  if (clarification) payload.clarification = clarification
  const resp = await sendMessage(currentSessionId.value, payload)
  if (resp.needs_clarification) {
    // 后端已建澄清任务：交出可交互的补充入口，而不是只丢一句“信息不足”
    pendingClarification.value = { taskId: resp.hitl_task_id, question: text }
    clarificationText.value = ''
    messages.value.push({
      role: 'system',
      content: resp.answer || '信息不足，请补充章节、人物或具体情节后继续提问。',
    })
  } else {
    pendingClarification.value = null
    messages.value.push({
      role: 'assistant',
      content: resp.answer || '',
      citations: resp.citations || [],
      cacheHit: !!resp.cache_hit,
      cacheChannel: resp.cache_channel || null,
      feedback: null,
      feedbackBusy: false,
    })
  }
}

async function send() {
  const text = question.value.trim()
  if (!text || sending.value) return
  sending.value = true
  question.value = ''
  messages.value.push({ role: 'user', content: text })
  scrollToBottom()
  try {
    await runQuestion(text)
  } catch (err) {
    messages.value.push({ role: 'system', content: `出错：${err.message}` })
  } finally {
    sending.value = false
    scrollToBottom()
  }
}

// 提交澄清：awaiting -> approved，并带着补充说明重跑原问题
async function submitClarify() {
  const pending = pendingClarification.value
  const extra = clarificationText.value.trim()
  if (!pending || !extra || clarificationBusy.value) return
  clarificationBusy.value = true
  try {
    // 服务端从断点继续（Command(resume=...)），直接把答案带回；
    // 只有服务端明确降级（resumed=false，例如旧任务或 checkpoint 已丢失）
    // 才回退到「带着补充说明重发原问题」的老路径。
    const submitted = pending.taskId ? await submitClarification(pending.taskId, extra) : null
    pendingClarification.value = null
    clarificationText.value = ''
    messages.value.push({ role: 'user', content: `补充说明：${extra}` })
    scrollToBottom()

    if (submitted && submitted.resumed) {
      messages.value.push({
        role: 'assistant',
        content: submitted.answer || '',
        citations: submitted.citations || [],
        feedback: null,
        feedbackBusy: false,
      })
      if (submitted.needs_clarification) {
        messages.value.push({
          role: 'system',
          content: '补充后信息仍然不足，请再补充章节、人物或具体情节。',
        })
      }
      await reloadHistory()
      return
    }

    sending.value = true
    await runQuestion(pending.question, extra)
    await reloadHistory()
  } catch (err) {
    messages.value.push({ role: 'system', content: `提交失败：${err.message}` })
  } finally {
    sending.value = false
    clarificationBusy.value = false
    scrollToBottom()
  }
}

// 放弃澄清：awaiting -> rejected
async function dismissClarify() {
  const pending = pendingClarification.value
  if (!pending || clarificationBusy.value) return
  clarificationBusy.value = true
  try {
    if (pending.taskId) await rejectClarification(pending.taskId)
    messages.value.push({ role: 'system', content: '已放弃本次澄清。' })
  } catch (err) {
    messages.value.push({ role: 'system', content: `操作失败：${err.message}` })
  } finally {
    pendingClarification.value = null
    clarificationText.value = ''
    clarificationBusy.value = false
  }
}

// 切换会话/刷新后恢复未处理的澄清任务
async function restorePendingClarification(sessionId) {
  pendingClarification.value = null
  clarificationText.value = ''
  if (!sessionId) return
  try {
    const tasks = await listHitlTasks(sessionId)
    const awaiting = (tasks || []).filter((t) => t.status === 'awaiting')
    if (!awaiting.length) return
    const last = awaiting[awaiting.length - 1]
    pendingClarification.value = { taskId: last.id, question: last.question }
  } catch {
    // 恢复失败不阻塞主流程
  }
}

// 记录用户对某条回答的评价；缓存来源由后端响应/消息 meta 透传，用于计算误命中率
async function submitVote(target, vote) {
  if (target.feedback || target.feedbackBusy) return
  target.feedbackBusy = true
  target.feedbackError = ''
  try {
    const index = messages.value.indexOf(target)
    let question = ''
    for (let i = index - 1; i >= 0; i -= 1) {
      if (messages.value[i].role === 'user') {
        question = messages.value[i].content
        break
      }
    }
    await submitFeedback({
      session_id: currentSessionId.value ? Number(currentSessionId.value) : null,
      question: question || '（未匹配到对应问题）',
      vote,
      cache_hit: !!target.cacheHit,
      cache_channel: target.cacheChannel || null,
    })
    target.feedback = vote
  } catch (err) {
    target.feedbackError = `提交失败：${err.message}`
  } finally {
    target.feedbackBusy = false
  }
}

function citationSource(c) {
  // chunk_id 形如 doc{document_id}-{index}，据此找到对应的书籍名
  const match = /^doc(\d+)-/.exec(c.chunk_id || '')
  const docId = match ? Number(match[1]) : null
  const doc = docId ? props.documents.find((d) => d.id === docId) : null
  const docName = doc ? doc.filename : docId ? `文档 ${docId}` : ''
  const chapter = c.chapter && c.chapter !== '正文' ? c.chapter : ''
  return [docName, chapter].filter(Boolean).join(' · ') || '引用'
}

const EXCERPT_PREVIEW_LEN = 60

function excerptPreview(text) {
  return text.length > EXCERPT_PREVIEW_LEN ? `${text.slice(0, EXCERPT_PREVIEW_LEN)}…` : text
}

function toggleExcerpt(c) {
  c.expanded = !c.expanded
}

async function scrollToBottom() {
  await nextTick()
  requestAnimationFrame(() => {
    const el = messageBox.value
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  })
}

function onMessagesScroll() {
  const el = messageBox.value
  if (!el) return
  stickToBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

watch(messages, () => {
  if (stickToBottom.value) scrollToBottom()
}, { deep: true })

onActivated(scrollToBottom)
onMounted(loadSessions)
</script>

<template>
  <div class="chat-layout">
    <aside class="chat-sidebar">
      <button class="new-chat" :disabled="sending" @click="createNewSession">
        <span class="plus">＋</span>
        新建对话
      </button>

      <div class="sidebar-section">
        <p class="field-label">历史会话</p>
        <ul v-if="sessions.length" class="session-list">
          <li
            v-for="s in sessions"
            :key="s.id"
            :class="{ active: s.id === currentSessionId }"
            @click="selectSession(s.id)"
          >
            <span class="session-row">
              <span class="session-title">{{ sessionLabel(s) }}</span>
              <button
                class="session-delete"
                title="删除会话"
                :disabled="sending"
                @click.stop="removeSession(s.id)"
              >
                ×
              </button>
            </span>
            <span class="session-time">{{ formatTime(s.created_at) }}</span>
          </li>
        </ul>
        <p v-else class="muted empty-inline">暂无会话</p>
      </div>
    </aside>

    <div class="chat-main">
      <header class="chat-header">
        <div class="chat-header-inner">
          <h2 class="chat-title">{{ currentTitle }}</h2>
        </div>
      </header>

      <div ref="messageBox" class="messages" @scroll="onMessagesScroll">
        <div class="messages-inner">
          <div v-if="!messages.length" class="empty-state">
            <div class="empty-icon">💬</div>
            <p class="empty-title">选择或新建会话，开始提问吧</p>
            <p class="empty-hint">例如：张三在本书中的事件时间线是怎样的？</p>
          </div>
          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <div class="msg-inner">
              <div class="msg-body">
                <span v-if="m.role === 'assistant'" class="meta">
                  AI 助手<span v-if="m.cacheHit" class="cache-tag" title="该回答直接复用了历史缓存">缓存</span>
                </span>
                <span class="content">{{ m.content }}</span>
                <div v-if="m.citations && m.citations.length" class="citations">
                  <div v-for="(c, j) in m.citations" :key="j" class="citation">
                    <span class="src">[{{ c.index || j + 1 }}] {{ citationSource(c) }}</span>
                    <div class="excerpt">
                      <span>
                        {{
                          c.excerpt.length > EXCERPT_PREVIEW_LEN && !c.expanded
                            ? excerptPreview(c.excerpt)
                            : c.excerpt
                        }}
                      </span>
                      <button
                        v-if="c.excerpt.length > EXCERPT_PREVIEW_LEN"
                        class="expand-btn"
                        @click="toggleExcerpt(c)"
                      >
                        {{ c.expanded ? '收起' : '展开' }}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              <div v-if="m.role === 'assistant' && m.content" class="msg-actions">
                <button
                  class="fb-btn"
                  :class="{ active: m.feedback === 'up' }"
                  :disabled="!!m.feedback || m.feedbackBusy"
                  title="回答有帮助"
                  @click="submitVote(m, 'up')"
                >
                  👍
                </button>
                <button
                  class="fb-btn"
                  :class="{ active: m.feedback === 'down' }"
                  :disabled="!!m.feedback || m.feedbackBusy"
                  title="回答有误（缓存答案被点踩会计入误命中率）"
                  @click="submitVote(m, 'down')"
                >
                  👎
                </button>
                <span v-if="m.feedbackError" class="fb-hint">{{ m.feedbackError }}</span>
                <span v-else-if="m.feedback" class="fb-hint">已反馈</span>
              </div>
            </div>
          </div>
          <div v-if="sending" class="msg assistant">
            <div class="msg-inner">
              <div class="msg-body typing">
                <span class="typing-text">思考中</span>
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
              </div>
            </div>
          </div>

          <div v-if="pendingClarification && !sending" class="msg clarify">
            <div class="msg-inner">
              <div class="clarify-card">
                <div class="clarify-head">
                  <span class="clarify-badge">待补充</span>
                  <span class="clarify-title">助手需要更多线索才能作答</span>
                </div>
                <p class="clarify-desc">
                  补充章节、人物或情节线索后，会带着你的补充重新检索原文；也可以直接放弃本次提问。
                </p>
                <p class="clarify-question">原问题：{{ pendingClarification.question }}</p>
                <textarea
                  v-model="clarificationText"
                  class="clarify-input"
                  rows="2"
                  placeholder="例如：第三章里关于张三的段落"
                  :disabled="clarificationBusy"
                  @keydown.enter.exact.prevent="submitClarify"
                ></textarea>
                <div class="clarify-actions">
                  <button
                    class="clarify-submit"
                    :disabled="!clarificationText.trim() || clarificationBusy"
                    @click="submitClarify"
                  >
                    {{ clarificationBusy ? '提交中…' : '提交补充并重新检索' }}
                  </button>
                  <button
                    class="clarify-dismiss"
                    :disabled="clarificationBusy"
                    @click="dismissClarify"
                  >
                    放弃
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="chat-input">
        <div class="input-inner">
          <div class="doc-picker">
            <span class="book-icon">📚</span>
            <span class="doc-picker-label">检索范围</span>
            <button
              type="button"
              class="scope-trigger"
              :class="{ open: scopeOpen }"
              @click.stop="scopeOpen = !scopeOpen"
            >
              <span class="scope-trigger-label">{{ scopeLabel }}</span>
              <span class="scope-trigger-arrow" :class="{ open: scopeOpen }">▾</span>
            </button>
            <span v-if="selectedDocIds.length > 1" class="multi-hint">对比 {{ selectedDocIds.length }} 本</span>
          </div>

          <div v-if="scopeOpen" class="scope-backdrop" @mousedown="scopeOpen = false"></div>
          <div v-if="scopeOpen" class="scope-menu">
            <div class="scope-menu-head">
              <button
                type="button"
                class="scope-option all"
                :class="{ active: selectedDocIds.length === 0 }"
                @click="selectedDocIds = []; scopeOpen = false"
              >
                <span class="check">{{ selectedDocIds.length === 0 ? '✓' : '' }}</span>
                <span class="name">全部书籍</span>
              </button>
              <span class="scope-count muted">{{ selectedDocIds.length }}/{{ documents.length }}</span>
            </div>
            <input
              v-model="scopeSearch"
              class="scope-search"
              type="text"
              placeholder="搜索书名…"
            />
            <div class="scope-list">
              <button
                v-for="doc in filteredDocs"
                :key="doc.id"
                type="button"
                class="scope-option"
                :class="{ active: selectedDocIds.includes(doc.id) }"
                :title="doc.filename"
                @click="toggleDoc(doc.id)"
              >
                <span class="check">{{ selectedDocIds.includes(doc.id) ? '✓' : '' }}</span>
                <span class="name">{{ doc.filename }}</span>
                <span class="meta muted">{{ doc.chunk_count }} 块</span>
              </button>
              <p v-if="!filteredDocs.length" class="scope-empty muted">没有匹配的书籍</p>
            </div>
            <p v-if="documents.length" class="scope-menu-foot muted">
              支持多选：跨书对比问答，引用会按书名分别标注
            </p>
          </div>
          <div class="input-row">
            <textarea
              v-model="question"
              rows="2"
              placeholder="向书籍提问，Enter 发送，Shift + Enter 换行"
              @keydown.enter.exact.prevent="send"
            ></textarea>
            <button class="send-btn" :disabled="sending || !question.trim()" @click="send">
              {{ sending ? '…' : '↑' }}
            </button>
          </div>
        </div>
        <p class="input-hint">回答基于已上传书籍内容；「检索范围」可多选，跨书自动对比并标注引用来源</p>
      </div>
    </div>
  </div>
</template>
