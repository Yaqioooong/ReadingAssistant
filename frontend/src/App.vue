<script setup>
import { ref, onMounted } from 'vue'
import { getDocuments } from './api.js'
import UploadPanel from './components/UploadPanel.vue'
import ChatPanel from './components/ChatPanel.vue'

const activeTab = ref('upload')
const documents = ref([])

async function refreshDocuments() {
  try {
    documents.value = await getDocuments()
  } catch {
    documents.value = []
  }
}

onMounted(refreshDocuments)
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
  </header>

  <main>
    <UploadPanel v-show="activeTab === 'upload'" :documents="documents" @uploaded="refreshDocuments" />
    <ChatPanel v-show="activeTab === 'chat'" :documents="documents" />
  </main>
</template>
