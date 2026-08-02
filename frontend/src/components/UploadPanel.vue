<script setup>
import { ref, computed } from 'vue'
import { uploadBook, formatTime } from '../api.js'

const props = defineProps({
  documents: { type: Array, default: () => [] },
})
const emit = defineEmits(['uploaded'])

const fileInput = ref(null)
const dragOver = ref(false)
const status = ref(null)

const empty = computed(() => props.documents.length === 0)

function pickFile() {
  fileInput.value?.click()
}

function onFileChange(event) {
  const file = event.target.files[0]
  if (file) handleFile(file)
  event.target.value = ''
}

async function handleFile(file) {
  status.value = { type: 'loading', text: `正在上传并解析「${file.name}」…` }
  try {
    const doc = await uploadBook(file)
    status.value = {
      type: 'ok',
      text: `解析完成：${doc.filename}（${doc.chunk_count} 个分块，document_id：${doc.id}）`,
    }
    emit('uploaded', doc)
  } catch (err) {
    status.value = { type: 'err', text: err.message || '上传失败' }
  }
}

function onDrop(event) {
  dragOver.value = false
  const file = event.dataTransfer.files[0]
  if (file) handleFile(file)
}
</script>

<template>
  <section class="card">
    <h2>上传电子书</h2>
    <div
      class="drop-zone"
      :class="{ dragover: dragOver }"
      role="button"
      tabindex="0"
      @click="pickFile"
      @keydown.enter="pickFile"
      @keydown.space.prevent="pickFile"
      @dragover.prevent="dragOver = true"
      @dragenter.prevent="dragOver = true"
      @dragleave.prevent="dragOver = false"
      @drop.prevent="onDrop"
    >
      <p class="drop-title">拖拽文件到此处，或点击选择</p>
      <p class="hint">支持 epub / pdf / docx / txt，旧版 .doc 请先转换</p>
    </div>
    <input
      ref="fileInput"
      type="file"
      accept=".epub,.pdf,.docx,.txt"
      hidden
      @change="onFileChange"
    />
    <div v-if="status" class="status" :class="status.type">{{ status.text }}</div>
  </section>

  <section class="card">
    <h2>已入库书籍</h2>
    <p v-if="empty" class="empty">暂无书籍，先上传一本试试</p>
    <table v-else class="table">
      <thead>
        <tr>
          <th>文件名</th>
          <th>标题</th>
          <th>作者</th>
          <th>分块数</th>
          <th>入库时间</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="doc in documents" :key="doc.id">
          <td>{{ doc.filename }}</td>
          <td>{{ doc.title || '-' }}</td>
          <td>{{ doc.author || '-' }}</td>
          <td>{{ doc.chunk_count }}</td>
          <td>{{ formatTime(doc.created_at) }}</td>
        </tr>
      </tbody>
    </table>
  </section>
</template>
