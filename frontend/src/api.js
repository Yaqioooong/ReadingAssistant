// FastAPI 后端接口封装
async function parseResponse(res) {
  if (res.ok) {
    if (res.status === 204) return null
    return res.json()
  }
  let detail = `请求失败（${res.status}）`
  try {
    const data = await res.json()
    if (typeof data.detail === 'string') detail = data.detail
  } catch {
    // 非 JSON 响应
  }
  throw new Error(detail)
}

function apiGet(path) {
  return fetch(path).then(parseResponse)
}

function apiPost(path, body) {
  return fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  }).then(parseResponse)
}

export const getDocuments = () => apiGet('/api/documents')
export const uploadBook = (file) => {
  const form = new FormData()
  form.append('file', file)
  return fetch('/api/documents/upload', { method: 'POST', body: form }).then(parseResponse)
}
export const createSession = () => apiPost('/api/sessions')
export const listSessions = () => apiGet('/api/sessions')
export const getHistory = (sessionId) => apiGet(`/api/sessions/${sessionId}/messages`)
export const sendMessage = (sessionId, payload) =>
  apiPost(`/api/sessions/${sessionId}/messages`, payload)

export function formatTime(iso) {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleString('zh-CN', { hour12: false })
  } catch {
    return iso
  }
}
