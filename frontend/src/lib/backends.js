const DEFAULT_BASE_URL = runtimeConfig().apiUrl || import.meta.env.VITE_API_URL || '/api'

export function createInitialSettings() {
  const saved = readSettings()
  return {
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || '',
    apiUrl: saved.apiUrl || DEFAULT_BASE_URL,
    adminToken: saved.adminToken || ''
  }
}

export function saveSettings(settings) {
  localStorage.setItem('goeuroops.frontend.settings', JSON.stringify(settings))
}

export function backendMeta(settings) {
  return {
    label: 'Python',
    baseUrl: normalizeBaseUrl(settings.apiUrl || DEFAULT_BASE_URL),
    port: '8000'
  }
}

export async function requestHealth(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/health')
}

export async function requestMonitor(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/monitor')
}

export async function requestSkills(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/skills')
}

export async function reloadSkills(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/skills/reload', { method: 'POST' })
}

export async function requestSkillDetail(settings, skillId) {
  return requestJson(backendMeta(settings).baseUrl, `/skills/${encodeURIComponent(skillId)}`)
}

export async function matchSkills(settings, body) {
  return requestJson(backendMeta(settings).baseUrl, '/skills/match', jsonRequest('POST', body))
}

export async function runSkillEvals(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/skills/evals')
}

export async function requestCatalog(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/catalog')
}

export async function requestLeads(settings, { status = '', type = '' } = {}) {
  const params = new URLSearchParams()
  if (status) params.set('status', status)
  if (type) params.set('type', type)
  const query = params.toString() ? `?${params}` : ''
  return requestJson(backendMeta(settings).baseUrl, `/leads${query}`, { headers: adminHeaders(settings) })
}

export async function updateLead(settings, leadId, patch) {
  const request = jsonRequest('PATCH', patch)
  request.headers = { ...request.headers, ...adminHeaders(settings) }
  return requestJson(backendMeta(settings).baseUrl, `/leads/${encodeURIComponent(leadId)}`, request)
}

export async function requestKnowledgeStats(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/stats')
}

export async function runEvaluation(settings, body = null) {
  return requestJson(backendMeta(settings).baseUrl, '/eval/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined
  })
}

export async function requestSearch(settings, query, topK = 5, domain = '') {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  if (domain) params.set('domain', domain)
  return requestJson(backendMeta(settings).baseUrl, `/search?${params}`, { method: 'POST' })
}

export async function streamChat(settings, message, { onToken, onDone, onError } = {}) {
  const meta = backendMeta(settings)
  const payload = buildChatPayload(settings, message)
  const response = await fetch(`${meta.baseUrl}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => '')
    throw new Error(`${response.status} ${response.statusText}: ${text}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let sepIndex
    while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sepIndex)
      buffer = buffer.slice(sepIndex + 2)
      const parsed = parseSseEvent(rawEvent)
      if (!parsed) continue
      if (parsed.event === 'token') {
        await onToken?.(parsed.data?.text || '')
      } else if (parsed.event === 'done') {
        await onDone?.(normalizeChatResponse(parsed.data))
      } else if (parsed.event === 'error') {
        await onError?.(new Error(parsed.data?.message || '流式请求失败'))
      }
    }
  }
}

function parseSseEvent(rawEvent) {
  let event = ''
  const dataLines = []
  for (const line of rawEvent.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!event) return null
  let data = null
  try {
    data = dataLines.length ? JSON.parse(dataLines.join('\n')) : null
  } catch {
    data = null
  }
  return { event, data }
}

export async function requestToolTrace(settings, requestId) {
  if (!requestId) return null
  const raw = await requestJson(backendMeta(settings).baseUrl, `/trace/tool/${encodeURIComponent(requestId)}`)
  return normalizeToolTraceResponse(raw)
}

export async function addKnowledge(settings, documents) {
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(settings, file, domain = '') {
  const form = new FormData()
  form.append('file', file)
  const query = domain ? `?domain=${encodeURIComponent(domain)}` : ''
  return requestJson(backendMeta(settings).baseUrl, `/knowledge/upload${query}`, {
    method: 'POST',
    body: form
  })
}

function buildChatPayload(settings, message) {
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conv_id: settings.conversationId || undefined
  }
}

function normalizeChatResponse(raw) {
  return {
    conversationId: raw.conv_id || raw.conversationId || '',
    requestId: raw.request_id || raw.requestId || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    intentGroup: raw.intent_group || raw.intentGroup || 'other',
    agentType: raw.agent_type || raw.agentType || '',
    agentTypes: raw.agent_types || raw.agentTypes || [],
    primaryAgent: raw.primary_agent || raw.primaryAgent || '',
    supportingAgents: raw.supporting_agents || raw.supportingAgents || [],
    routingReason: raw.routing_reason || raw.routingReason || '',
    routingConfidence: Number(raw.routing_confidence ?? raw.routingConfidence ?? 0),
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? raw.intentConfidence ?? 0),
    intentSourceScores: raw.intent_source_scores || raw.intentSourceScores || {},
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    skillsApplied: raw.skills_applied || raw.skillsApplied || [],
    ragGate: raw.rag_gate || raw.ragGate || {},
    toolsUsed: raw.tools_used || raw.toolsUsed || [],
    raw
  }
}

function normalizeToolTraceResponse(raw) {
  const trace = raw?.trace || {}
  return {
    requestId: raw?.request_id || raw?.requestId || '',
    found: Boolean(raw?.found),
    trace: {
      ...trace,
      toolsUsed: trace.tools_used || trace.toolsUsed || [],
      toolCalls: trace.tool_calls || trace.toolCalls || []
    },
    raw
  }
}

function jsonRequest(method, body) {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {})
  }
}

function adminHeaders(settings) {
  return settings.adminToken ? { 'X-Admin-Token': settings.adminToken } : {}
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem('goeuroops.frontend.settings') || '{}')
  } catch {
    return {}
  }
}

function runtimeConfig() {
  if (typeof window === 'undefined') return {}
  return window.__GOEUROOPS_CONFIG__ || {}
}
