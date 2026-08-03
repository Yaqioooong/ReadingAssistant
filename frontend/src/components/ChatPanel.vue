<script setup>
import { ref, computed, nextTick, onMounted } from 'vue'
import {
  createSession,
  listSessions,
  getHistory,
  sendMessage,
  formatTime,
} from '../api.js'

const props = defineProps({
  documents: { type: Array, default: () => [] },
})

const sessions = ref([])
const currentSessionId = ref(null)
const messages = ref([])
const question = ref('')
const filterDoc = ref('')
const sending = ref(false)
const messageBox = ref(null)

const currentTitle = computed(() => {
  const session = sessions.value.find((item) => item.id === currentSessionId.value)
  return session ? sessionLabel(session) : '新的对话'
})

function sessionLabel(session) {
  return `会话 ${session.id.slice(0, 8)}`
}

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
    await loadSessions()
    scrollToBottom()
  } catch (err) {
    messages.value = [{ role: 'system', content: `出错：${err.message}` }]
  }
}

async function selectSession(id) {
  currentSessionId.value = id
  try {
    const history = await getHistory(id)
    messages.value = history.map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : m.role === 'user' ? 'user' : 'system',
      content: m.content,
      citations: m.meta?.citations || [],
    }))
  } catch (err) {
    messages.value = [{ role: 'system', content: `出错：${err.message}` }]
  }
}

async function send() {
  const text = question.value.trim()
  if (!text || sending.value) return
  sending.value = true
  try {
    if (!currentSessionId.value) {
      const { session_id } = await createSession()
      currentSessionId.value = session_id
      await loadSessions()
    }
    messages.value.push({ role: 'user', content: text })
    question.value = ''
    const documentIds = filterDoc.value ? [filterDoc.value] : []
    const resp = await sendMessage(currentSessionId.value, {
      question: text,
      document_ids: documentIds,
    })
    if (resp.needs_clarification) {
      messages.value.push({
        role: 'system',
        content: resp.answer || '信息不足，请补充章节、人物或具体情节后继续提问。',
      })
    } else {
      messages.value.push({
        role: 'assistant',
        content: resp.answer || '',
        citations: resp.citations || [],
      })
    }
  } catch (err) {
    messages.value.push({ role: 'system', content: `出错：${err.message}` })
  } finally {
    sending.value = false
    scrollToBottom()
  }
}

function citationSource(c) {
  return [c.chapter, c.page ? `第 ${c.page} 页` : null].filter(Boolean).join(' · ') || '引用'
}

async function scrollToBottom() {
  await nextTick()
  if (messageBox.value) messageBox.value.scrollTop = messageBox.value.scrollHeight
}

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
            <span class="session-title">{{ sessionLabel(s) }}</span>
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
          <label class="filter">
            <span class="filter-label">检索范围</span>
            <select v-model="filterDoc" class="select">
              <option value="">全部书籍</option>
              <option v-for="doc in documents" :key="doc.id" :value="doc.id">
                {{ doc.filename }}
              </option>
            </select>
          </label>
        </div>
      </header>

      <div ref="messageBox" class="messages">
        <div class="messages-inner">
          <div v-if="!messages.length" class="empty-state">
            <div class="empty-icon">💬</div>
            <p class="empty-title">选择或新建会话，开始提问吧</p>
            <p class="empty-hint">例如：张三在本书中的事件时间线是怎样的？</p>
          </div>
          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <div class="msg-inner">
              <div class="msg-body">
                <span v-if="m.role === 'assistant'" class="meta">AI 助手</span>
                <span class="content">{{ m.content }}</span>
                <div v-if="m.citations && m.citations.length" class="citations">
                  <div v-for="(c, j) in m.citations" :key="j" class="citation">
                    <span class="src">{{ citationSource(c) }}</span>
                    <div class="excerpt">{{ c.excerpt }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="chat-input">
        <div class="input-inner">
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
        <p class="input-hint">回答基于已上传书籍内容，可在右上角指定检索范围</p>
      </div>
    </div>
  </div>
</template>
