<script setup>
import { ref, onMounted } from 'vue'
import { getDocuments } from './api.js'
import UploadPanel from './components/UploadPanel.vue'
import ChatPanel from './components/ChatPanel.vue'

const activeTab = ref('upload')
const documents = ref([])
const theme = ref('dark')
const THEME_KEY = 'reading-assistant-theme'

function setTheme(value) {
  theme.value = value
  document.documentElement.dataset.theme = value
}

function toggleTheme() {
  const next = theme.value === 'dark' ? 'light' : 'dark'
  setTheme(next)
  try {
    localStorage.setItem(THEME_KEY, next)
  } catch {
    // 忽略存储不可用的情况
  }
}

async function refreshDocuments() {
  try {
    documents.value = await getDocuments()
  } catch {
    documents.value = []
  }
}

onMounted(() => {
  let initial = null
  try {
    initial = localStorage.getItem(THEME_KEY)
  } catch {
    // 忽略存储不可用的情况
  }
  if (initial !== 'light' && initial !== 'dark') {
    initial = window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
  }
  setTheme(initial)

  const media = window.matchMedia?.('(prefers-color-scheme: light)')
  media?.addEventListener?.('change', (event) => {
    let saved = null
    try {
      saved = localStorage.getItem(THEME_KEY)
    } catch {
      // 忽略存储不可用的情况
    }
    if (!saved) setTheme(event.matches ? 'light' : 'dark')
  })

  refreshDocuments()
})
</script>

<template>
  <header class="topbar">
    <div class="brand">📚 阅读 Agent</div>
    <nav class="tabs">
      <button class="tab" :class="{ active: activeTab === 'upload' }" @click="activeTab = 'upload'">
        上传解析
      </button>
      <button class="tab" :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">
        聊天问答
      </button>
    </nav>
    <button
      class="theme-toggle"
      :title="theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'"
      @click="toggleTheme"
    >
      <span class="theme-icon">{{ theme === 'dark' ? '🌙' : '☀️' }}</span>
      <span class="theme-label">{{ theme === 'dark' ? '深色' : '浅色' }}</span>
    </button>
  </header>

  <main :class="{ padded: activeTab === 'upload' }">
    <KeepAlive>
      <UploadPanel
        v-if="activeTab === 'upload'"
        :documents="documents"
        @uploaded="refreshDocuments"
      />
      <ChatPanel v-else :documents="documents" />
    </KeepAlive>
  </main>
</template>
