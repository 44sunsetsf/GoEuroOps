<template>
  <main :class="['app-shell', `app-shell-${activeView}`, { 'app-shell-scroll': activeView !== 'chat' }]">
    <header class="topbar">
      <a class="brand" href="#" aria-label="指北首页" @click.prevent="activeView = 'chat'">
        <span class="brand-mark">北</span>
        <span class="brand-copy">
          <span class="brand-name">指北 · True North</span>
          <span class="brand-sub">留学运营助手控制台</span>
        </span>
      </a>

      <nav class="view-nav" aria-label="工作区">
        <button
          v-for="view in VIEWS"
          :key="view.id"
          :class="{ active: activeView === view.id }"
          @click="activeView = view.id"
        >{{ view.label }}</button>
      </nav>

      <div class="topbar-tools">
        <span class="environment-pill">
          <i :class="healthOk ? 'online' : 'offline'"></i>
          {{ healthOk ? '服务在线' : '服务离线' }}
        </span>
        <a class="docs-link" :href="docsUrl" target="_blank" rel="noreferrer">API 文档</a>
        <button class="avatar-button" title="当前用户">{{ userInitial }}</button>
      </div>
    </header>

    <div v-if="toast" class="toast" role="status">{{ toast }}</div>

    <section v-if="activeView === 'chat'" class="page page-chat">
      <div class="page-heading">
        <div class="heading-copy">
          <h1>指北留学运营助手</h1>
          <p>这里是「指北」留学工作室的内部调试台——测试助手怎么回答瑞典、德国、荷兰、芬兰、丹麦英语授课 CS 硕士的常见问题、介绍我们的选校和文书服务，并判断什么时候该转给人工顾问。</p>
        </div>
        <div class="heading-actions">
          <span class="session-label">{{ settings.conversationId || '新会话' }}</span>
          <button class="quiet-button" @click="clearConversation">清空</button>
        </div>
      </div>

      <div class="chat-layout">
        <section class="chat-stage">
          <div class="stage-bar">
            <div class="stage-context">
              <span class="context-dot"></span>
              <span>{{ currentBackend.baseUrl }}</span>
            </div>

            <span>{{ messages.length }} 条消息</span>
          </div>

          <div class="messages" ref="messageList">
            <article v-for="item in messages" :key="item.id" :class="['message', item.role]">
              <div class="message-meta">
                <span>{{ item.role === 'user' ? '你' : '指北助手' }}</span>
                <small v-if="item.streaming">生成中…</small>
                <small v-else-if="item.meta">{{ item.meta }}</small>
              </div>
              <div v-if="item.chips?.length" class="message-chips">
                <span
                  v-for="chip in item.chips"
                  :key="chip.text"
                  :class="['chip', chip.kind && `chip-${chip.kind}`]"
                  :title="chip.title || ''"
                >{{ chip.text }}</span>
              </div>
              <div v-if="item.streaming && !item.content" class="thinking-dots">
                <span></span><span></span><span></span>
              </div>
              <div
                v-else
                class="message-markdown"
                :class="{ streaming: item.streaming }"
                v-html="renderMarkdown(item.content)"
              ></div>
              <div v-if="item.trace" class="message-trace">
                <div class="trace-head">
                  <span>工具调用</span>
                  <small v-if="item.trace.requestId">#{{ item.trace.requestId }}</small>
                </div>
                <div v-if="item.trace.toolCalls?.length" class="trace-calls">
                  <details v-for="(call, index) in item.trace.toolCalls" :key="`${item.id}-${index}`" open>
                    <summary>
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.success ? '成功' : '失败' }}</span>
                    </summary>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </details>
                </div>
                <div v-else class="trace-empty-block">
                  <p>本次请求已生成 trace，但没有可展示的工具输入。</p>
                  <p v-if="item.trace.toolsUsed?.length" class="trace-note">已调用：{{ item.trace.toolsUsed.join(' · ') }}</p>
                </div>
              </div>
            </article>

            <div v-if="messages.length === 0" class="empty-state">
              <div class="empty-symbol">✦</div>
              <h2>从一个学生会问的问题开始</h2>
              <p>下面的快捷问题覆盖了咨询、报价、边界和售后几类场景，你也可以直接输入自己的测试用例。</p>
              <div class="starter-prompts">
                <button v-for="prompt in STARTER_PROMPTS" :key="prompt.label" @click="usePrompt(prompt.text)">{{ prompt.label }}</button>
              </div>
            </div>
          </div>

          <form class="composer" @submit.prevent="sendMessage">
            <textarea
              v-model="draft"
              rows="3"
              placeholder="输入消息..."
              @keydown.meta.enter.prevent="sendMessage"
              @keydown.ctrl.enter.prevent="sendMessage"
            ></textarea>
            <div class="composer-bottom">
              <span>⌘ / Ctrl + Enter 发送</span>
              <button type="submit" :disabled="busy || !draft.trim()">{{ busy ? '处理中' : '发送' }}</button>
            </div>
          </form>
        </section>

        <aside class="chat-sidebar" ref="sidebarRef">
          <div class="chat-sidebar-scroll">
            <section class="side-card session-card">
              <div class="card-heading">
                <div>
                  <h2>会话信息</h2>
                </div>
                <span class="status-copy muted">{{ settings.conversationId ? '已启用' : '新会话' }}</span>
              </div>
              <div class="session-grid">
                <div>
                  <span>会话 ID</span>
                  <strong>{{ settings.conversationId || '自动生成' }}</strong>
                </div>
                <div>
                  <span>用户 ID</span>
                  <strong>{{ settings.userId || 'anonymous' }}</strong>
                </div>
              </div>
            </section>

            <section class="side-card connection-card">
              <div class="card-heading">
                <div>
                  <h2>连接配置</h2>
                </div>
                <span class="status-copy" :class="healthOk ? 'success' : 'muted'">{{ healthLabel }}</span>
              </div>

              <label>
                <span>用户 ID</span>
                <input v-model="settings.userId" @change="persist" placeholder="u1001" />
              </label>
              <label>
                <span>会话 ID</span>
                <input v-model="settings.conversationId" @change="persist" placeholder="自动生成" />
              </label>
              <label>
                <span>Admin Token（线索面板）</span>
                <input v-model="settings.adminToken" type="password" @change="persist" placeholder="未设置 GOEUROOPS_ADMIN_TOKEN 时留空" />
              </label>
              <div class="side-actions">
                <button @click="checkHealth">检查连接</button>
                <button class="quiet-button" @click="refreshConsole">刷新</button>
              </div>
            </section>

            <section class="side-card trace-card">
              <div class="card-heading">
                <div>
                  <h2>最近一次请求</h2>
                </div>
                <span class="trace-status" :class="lastResponse ? 'has-data' : ''"></span>
              </div>

              <div v-if="lastResponse" class="trace-body">
                <div class="latency">
                  <span>响应耗时</span>
                  <strong>{{ displayedLatency }}<small> ms</small></strong>
                </div>
                <dl class="detail-list">
                  <div><dt>主 Agent</dt><dd>{{ lastResponse.primaryAgent || lastResponse.agentType || '-' }}</dd></div>
                  <div><dt>意图</dt><dd>{{ lastResponse.intent || '-' }}</dd></div>
                  <div><dt>置信度</dt><dd>{{ formatPercent(lastResponse.routingConfidence) }}</dd></div>
                  <div><dt>RAG 门控</dt><dd :class="lastResponse.ragGate?.mode === 'prefetch' ? 'success' : 'muted'">{{ ragLabel(lastResponse.ragGate) }}</dd></div>
                  <div><dt>知识库</dt><dd :class="lastResponse.knowledgeUsed ? 'success' : 'muted'">{{ lastResponse.knowledgeUsed ? '已使用' : '未使用' }}</dd></div>
                  <div><dt>Skills</dt><dd>{{ lastResponse.skillsApplied?.map((s) => s.id).join('、') || '无' }}</dd></div>
                  <div><dt>转顾问</dt><dd :class="lastResponse.escalated ? 'danger' : 'muted'">{{ lastResponse.escalated ? '是' : '否' }}</dd></div>
                </dl>
                <p v-if="lastResponse.ragGate?.reason" class="routing-reason">{{ lastResponse.ragGate.reason }}</p>
                <p v-if="lastResponse.routingReason" class="routing-reason">{{ lastResponse.routingReason }}</p>
                <div v-if="lastTrace?.trace" class="trace-call-list">
                  <div class="trace-call-title">工具调用</div>
                  <div v-for="(call, index) in lastTrace.trace.toolCalls" :key="`${call.tool_use_id || index}`" class="trace-call-item">
                    <div class="trace-call-meta">
                      <strong>{{ call.tool_name || 'unknown_tool' }}</strong>
                      <span>{{ call.latency_ms || 0 }} ms</span>
                    </div>
                    <pre>{{ formatJson(call.input || {}) }}</pre>
                  </div>
                  <div v-if="!lastTrace.trace.toolCalls?.length" class="trace-empty-block">
                    <p>这次 trace 没有记录到工具输入。</p>
                    <p v-if="lastTrace.trace.toolsUsed?.length" class="trace-note">已调用：{{ lastTrace.trace.toolsUsed.join(' · ') }}</p>
                  </div>
                </div>
              </div>
              <p v-else class="side-empty">发送消息后，这里会显示 Agent 路由、意图和耗时。</p>
            </section>

            <section class="side-card monitor-card">
              <div class="card-heading">
                <div>
                  <h2>运行状态</h2>
                </div>
                <button class="link-button" @click="loadMonitor">刷新</button>
              </div>
              <div class="mini-stats">
                <div><strong>{{ totalRequests }}</strong><span>请求</span></div>
                <div><strong>{{ agentCount }}</strong><span>Agent</span></div>
                <div><strong>{{ activeAlerts.length }}</strong><span>告警</span></div>
              </div>
              <div v-if="activeAlerts.length" class="alert-note">{{ activeAlerts[0].detail || activeAlerts[0].title }}</div>
              <p v-else class="healthy-note">当前没有活跃告警。</p>
            </section>
          </div>
        </aside>
      </div>
    </section>

    <section v-else-if="activeView === 'knowledge'" class="page page-knowledge">
      <div class="page-heading">
        <div class="heading-copy">
          <h1>知识库</h1>
          <p>
            助手回答时引用的参考资料。服务、价格、政策和五国资料由业务目录自动生成（种子版本 <code>{{ knowledgeStats.seed_version || '-' }}</code>），
            这里导入的是补充文档。向量模型：<code>{{ knowledgeStats.embedding_model || '-' }}</code>
          </p>
        </div>
        <div class="count-display"><strong>{{ knowledgeCount }}</strong><span>chunks · {{ knowledgeStats.documents ?? '-' }} 篇</span></div>
      </div>
      <div v-if="knowledgeStats.by_domain" class="domain-strip">
        <span v-for="(count, domain) in knowledgeStats.by_domain" :key="domain" class="chip">{{ domainLabel(domain) }} {{ count }}</span>
        <span v-for="(count, source) in knowledgeStats.by_source" :key="source" class="chip chip-accent">{{ source === 'seed' ? '业务目录生成' : '手动导入' }} {{ count }}</span>
      </div>

      <div class="knowledge-layout">
        <section class="workspace-card search-workspace">
          <div class="card-heading">
            <div><h2>检索知识</h2></div>
            <code>POST /search</code>
          </div>
          <div class="search-line">
            <input v-model="searchQuery" placeholder="例如：瑞典申请什么时候截止" @keydown.enter="searchKnowledge" />
            <select v-model="searchDomain" class="domain-select">
              <option v-for="item in DOMAIN_OPTIONS" :key="item.value" :value="item.value">{{ item.label }}</option>
            </select>
            <button @click="searchKnowledge" :disabled="busy || !searchQuery.trim()">搜索</button>
          </div>
          <div v-if="searchResults.length" class="result-list">
            <article v-for="(item, index) in searchResults" :key="item.id || item.title || index" class="result-item">
              <span class="result-number">{{ String(index + 1).padStart(2, '0') }}</span>
              <div>
                <div class="result-title">
                  <strong>{{ item.title || '未命名文档' }}</strong>
                  <small>{{ domainLabel(item.domain) }} · 相关度 {{ item.relevance ?? '-' }}/10 · 向量 {{ item.vector_score ?? item.score ?? '-' }}</small>
                </div>
                <p>{{ item.content }}</p>
              </div>
            </article>
          </div>
          <div v-else class="workspace-empty">输入学生可能问的问题开始搜索（完整链路：查询改写 → 并行召回 → 去重 → LLM 重排）。</div>
        </section>

        <section class="workspace-card import-workspace">
          <div class="card-heading">
            <div><h2>添加知识</h2></div>
            <code>ChromaDB</code>
          </div>
          <label><span>标题</span><input v-model="docTitle" placeholder="KTH 2027 申请更新" /></label>
          <label>
            <span>归属</span>
            <select v-model="docDomain">
              <option v-for="item in DOMAIN_OPTIONS.filter((d) => d.value)" :key="item.value" :value="item.value">{{ item.label }}</option>
            </select>
          </label>
          <label><span>内容</span><textarea v-model="docContent" rows="7" placeholder="输入院校更新、FAQ 或内部说明"></textarea></label>
          <div class="side-actions">
            <button @click="submitKnowledge" :disabled="busy || !docTitle.trim() || !docContent.trim()">添加文档</button>
            <label class="upload-button">上传文件<input type="file" accept=".txt,.md,.json" @change="handleUpload" /></label>
          </div>
        </section>
      </div>

    </section>

    <LeadsView v-else-if="activeView === 'leads'" :settings="settings" @toast="showToast" />
    <CatalogView v-else-if="activeView === 'catalog'" :settings="settings" />
    <SkillsView v-else-if="activeView === 'skills'" :settings="settings" @toast="showToast" />

    <section v-else class="page page-evaluation">
      <div class="page-heading">
        <div class="heading-copy">
          <h1>评测助手</h1>
          <p>意图识别准确率、Skill 路由准确率、按场景校准的 LLM-as-Judge 五维评分（含边界合规）、工具调用检查和回归对比。</p>
        </div>
        <div class="heading-actions">
          <label class="toggle">
            <input type="checkbox" v-model="compareRagGate" />
            <span>RAG 门控对照实验（多调用一轮 LLM）</span>
          </label>
          <button @click="runEvaluation" :disabled="busy">{{ busy ? '运行中...' : '运行评测' }}</button>
        </div>
      </div>

      <div v-if="evalData" class="evaluation-content">
        <div class="evaluation-summary">
          <div class="score-hero"><span>Pass rate</span><strong>{{ formatPercent(evalData.pass_rate) }}</strong><small>{{ evalData.passed }} / {{ evalData.total }} cases passed</small></div>
          <div><span>通过</span><strong>{{ evalData.passed }}</strong></div>
          <div><span>总数</span><strong>{{ evalData.total }}</strong></div>
          <div><span>回归</span><strong :class="evalData.regressions?.length ? 'danger' : 'success'">{{ evalData.regressions?.length || 0 }}</strong></div>
        </div>
        <div class="evaluation-layout">
          <section class="workspace-card">
            <div class="card-heading"><div><h2>平均评分</h2></div></div>
            <div class="score-list">
              <div v-for="(value, key) in evalData.avg_scores" :key="key"><span>{{ SCORE_LABELS[key] || key }}</span><i><b :style="{ width: `${Math.min(Number(value), 1) * 100}%` }"></b></i><strong>{{ Number(value).toFixed(2) }}</strong></div>
            </div>
          </section>
          <section class="workspace-card">
            <div class="card-heading"><div><h2>优化建议</h2></div></div>
            <div v-if="evalData.recommendations?.length" class="recommendations"><p v-for="(item, index) in evalData.recommendations" :key="index">{{ item }}</p></div>
            <div v-else class="workspace-empty">本次评测没有返回额外建议。</div>
          </section>
        </div>
        <section v-if="ragAb" class="workspace-card">
          <div class="card-heading"><div><h2>RAG 门控对照实验</h2></div></div>
          <p class="hint">{{ ragAb.detail }}</p>
          <div class="ab-table">
            <div class="ab-row ab-head"><span>问题</span><span>门控</span><span>模型自行检索</span></div>
            <div v-for="row in ragAb.metadata.cases" :key="row.question" class="ab-row">
              <span>{{ row.question }}</span>
              <span>{{ row.gated.latency_ms }} ms · 工具 {{ row.gated.tool_calls }} 次 · {{ ragModeText(row.gated.rag_mode) }}{{ row.gated.prefetched ? ` ${row.gated.prefetched} 条` : '' }}</span>
              <span>{{ row.baseline.latency_ms }} ms · 工具 {{ row.baseline.tool_calls }} 次</span>
            </div>
          </div>
        </section>
        <section class="workspace-card">
          <div class="card-heading"><div><h2>对话用例明细</h2></div></div>
          <details v-for="item in dialogResults" :key="item.test_id" class="dialog-result">
            <summary>
              <span :class="item.passed ? 'success' : 'danger'">{{ item.passed ? '通过' : '未通过' }}</span>
              <strong>{{ item.metadata.question }}</strong>
              <small>{{ Number(item.scores.overall).toFixed(2) }}</small>
            </summary>
            <p v-if="item.metadata.expected_behavior" class="hint">期望：{{ item.metadata.expected_behavior }}</p>
            <p v-if="item.metadata.judge_comment && item.metadata.judge_comment !== '无'" class="hint">评审意见：{{ item.metadata.judge_comment }}</p>
            <div class="signal-chips">
              <span v-for="dim in ['relevance', 'accuracy', 'completeness', 'helpfulness', 'compliance']" :key="dim" class="chip">{{ SCORE_LABELS[dim] }} {{ Number(item.scores[dim] ?? 0).toFixed(2) }}</span>
              <span v-if="item.scores.tool_check != null" :class="['chip', item.scores.tool_check ? 'chip-accent' : 'chip-danger']">工具 {{ item.scores.tool_check ? '✓' : '✗' }}</span>
              <span class="chip">{{ label(AGENT_LABELS, item.metadata.agent_type) }}</span>
              <span v-for="skill in item.metadata.skills_applied || []" :key="skill" class="chip chip-spark">{{ skill }}</span>
            </div>
            <div class="message-markdown eval-answer" v-html="renderMarkdown(item.metadata.response || '')"></div>
          </details>
        </section>
      </div>
      <div v-else class="evaluation-empty"><div class="empty-symbol">◎</div><h2>还没有评测结果</h2><p>点击右上角运行一次评测。</p></div>
    </section>
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import CatalogView from './components/CatalogView.vue'
import LeadsView from './components/LeadsView.vue'
import SkillsView from './components/SkillsView.vue'
import {
  addKnowledge,
  backendMeta,
  createInitialSettings,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  requestToolTrace,
  runEvaluation as requestEvaluation,
  saveSettings,
  streamChat,
  uploadKnowledge
} from './lib/backends'
import { renderMarkdown } from './lib/markdown'
import { AGENT_LABELS, DOMAIN_OPTIONS, INTENT_LABELS, RAG_MODE_LABELS, label } from './lib/labels'

const VIEWS = [
  { id: 'chat', label: '对话' },
  { id: 'leads', label: '线索' },
  { id: 'catalog', label: '服务价目' },
  { id: 'skills', label: 'Skills' },
  { id: 'knowledge', label: '知识库' },
  { id: 'evaluation', label: '评测' }
]

const STARTER_PROMPTS = [
  { label: '瑞典 CS 硕士', text: '瑞典的英语授课计算机硕士怎么申请？什么时候截止？' },
  { label: '服务和价格', text: '你们都提供哪些服务？大概多少钱？' },
  { label: '算个报价', text: '我想要选校全案加 3 个项目的文书套餐，9 月底前签约，是老学员介绍的，一共多少钱？' },
  { label: '能不能上 KTH', text: '我均分 85、雅思 7，你直接告诉我能不能上 KTH' },
  { label: '退款规则', text: '全程陪跑交了定金，服务还没开始，可以退吗？' },
  { label: '留联系方式', text: '我想预约一次咨询，微信是 wang_2027，叫我小王就行' }
]

const SCORE_LABELS = {
  relevance: '相关性',
  accuracy: '准确性',
  completeness: '完整性',
  helpfulness: '有用性',
  compliance: '边界合规',
  intent_accuracy: '意图准确率',
  skill_routing_accuracy: 'Skill 路由',
  tool_accuracy: '工具调用'
}

const settings = reactive(createInitialSettings())
const activeView = ref('chat')
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const healthOk = ref(false)
const healthLabel = ref('未检查')
const statusText = ref('')
const knowledgeCount = ref('-')
const knowledgeStats = ref({})
const searchQuery = ref('瑞典申请什么时候截止')
const searchDomain = ref('')
const searchResults = ref([])
const docTitle = ref('KTH 2027 申请季更新')
const docDomain = ref('consulting')
const docContent = ref('示例：KTH 2027 秋季入学的国际轮申请预计 10 月中旬开放，以官网公告为准。')
const compareRagGate = ref(false)
const messageList = ref(null)
const sidebarRef = ref(null)
const monitorData = ref({ agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] })
const lastResponse = ref(null)
const lastTrace = ref(null)
const displayedLatency = ref('-')
let latencyRaf = null
const evalData = ref(null)
const toast = ref('')
let toastTimer
let messageSequence = 0
let sidebarObserver

const currentBackend = computed(() => backendMeta(settings))
const docsUrl = computed(() => `${currentBackend.value.baseUrl}/docs`)
const userInitial = computed(() => (settings.userId || 'U').slice(0, 1).toUpperCase())
const activeAlerts = computed(() => monitorData.value.active_alerts || [])
const agentCount = computed(() => Object.keys(monitorData.value.agent_stats || {}).length)
const totalRequests = computed(() => Object.values(monitorData.value.agent_stats || {}).reduce((sum, item) => sum + Number(item.total || 0), 0))
const dialogResults = computed(() => (evalData.value?.results || []).filter((r) => r.scores?.overall != null))
const ragAb = computed(() => (evalData.value?.results || []).find((r) => r.test_id === 'rag_gate_ab'))

watch(() => settings.conversationId, persist)
watch(() => lastResponse.value?.latencyMs, (target) => {
  if (typeof target === 'number' && target > 0) animateLatency(target)
})
onMounted(() => {
  refreshConsole()
  updateSidebarHeight()
  if (typeof ResizeObserver !== 'undefined') {
    sidebarObserver = new ResizeObserver(updateSidebarHeight)
    if (sidebarRef.value) sidebarObserver.observe(sidebarRef.value)
  }
  window.addEventListener('resize', updateSidebarHeight)
})

onBeforeUnmount(() => {
  sidebarObserver?.disconnect?.()
  window.removeEventListener('resize', updateSidebarHeight)
  if (latencyRaf) cancelAnimationFrame(latencyRaf)
})

function animateLatency(target) {
  if (latencyRaf) cancelAnimationFrame(latencyRaf)
  const duration = 550
  const start = performance.now()
  const step = (now) => {
    const progress = Math.min(1, (now - start) / duration)
    const eased = 1 - Math.pow(1 - progress, 3)
    displayedLatency.value = Math.round(target * eased)
    if (progress < 1) {
      latencyRaf = requestAnimationFrame(step)
    } else {
      displayedLatency.value = target
      latencyRaf = null
    }
  }
  latencyRaf = requestAnimationFrame(step)
}

function persist() { saveSettings(settings) }

function updateSidebarHeight() {
  const sidebar = sidebarRef.value
  if (!sidebar) return
  const rect = sidebar.getBoundingClientRect()
  const height = Math.max(320, Math.floor(rect.height))
  sidebar.style.setProperty('--sidebar-height', `${height}px`)
}

async function refreshConsole() {
  await Promise.allSettled([checkHealth(), loadStats(), loadMonitor()])
}

async function checkHealth() {
  try {
    const data = await requestHealth(settings)
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.status || 'ok'
    statusText.value = JSON.stringify(data, null, 2)
  } catch (error) {
    healthOk.value = false
    healthLabel.value = '不可用'
    statusText.value = error.message
  }
}

async function loadStats() {
  try {
    const data = await requestKnowledgeStats(settings)
    knowledgeStats.value = data || {}
    knowledgeCount.value = data.total_chunks ?? data.totalChunks ?? '-'
  } catch {
    knowledgeCount.value = '-'
    knowledgeStats.value = {}
  }
}

async function loadMonitor() {
  try {
    monitorData.value = await requestMonitor(settings)
  } catch {
    monitorData.value = { agent_stats: {}, tool_stats: {}, active_alerts: [], suggestions: [] }
  }
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content || busy.value) return
  messages.value.push({ id: createMessageId(), role: 'user', content })
  draft.value = ''
  busy.value = true

  const assistantMessage = reactive({
    id: createMessageId(),
    role: 'assistant',
    content: '',
    meta: '',
    trace: null,
    chips: [],
    streaming: true
  })
  messages.value.push(assistantMessage)
  scrollToBottom()

  try {
    await streamChat(settings, content, {
      onToken(text) {
        assistantMessage.content += text
        scrollToBottom()
      },
      async onDone(response) {
        assistantMessage.streaming = false
        if (response.conversationId && !settings.conversationId) {
          settings.conversationId = response.conversationId
          persist()
        }
        lastResponse.value = response
        lastTrace.value = await loadToolTrace(response.requestId)
        assistantMessage.trace = lastTrace.value?.trace || null
        assistantMessage.meta = [
          label(INTENT_LABELS, response.intent),
          label(AGENT_LABELS, response.primaryAgent || response.agentType),
          response.escalated ? '已转顾问' : ''
        ].filter(Boolean).join(' · ')
        assistantMessage.chips = messageChips(response)
        await loadMonitor()
      },
      onError(error) {
        assistantMessage.streaming = false
        if (!assistantMessage.content) assistantMessage.content = error.message
        assistantMessage.meta = '请求失败'
      }
    })
  } catch (error) {
    assistantMessage.streaming = false
    if (!assistantMessage.content) assistantMessage.content = error.message
    assistantMessage.meta = '请求失败'
  } finally {
    busy.value = false
    scrollToBottom()
  }
}

function scrollToBottom() {
  nextTick(() => {
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  })
}

function usePrompt(prompt) { draft.value = prompt }

function ragLabel(gate) {
  if (!gate?.mode) return '-'
  const base = label(RAG_MODE_LABELS, gate.mode)
  return gate.mode === 'prefetch' ? `${base} ${gate.prefetched || 0} 条` : base
}

function ragModeText(mode) { return label(RAG_MODE_LABELS, mode) }

function domainLabel(domain) {
  return DOMAIN_OPTIONS.find((item) => item.value === domain)?.label || domain || '共享'
}

function messageChips(response) {
  const chips = []
  if (response.ragGate?.mode) {
    chips.push({
      text: ragLabel(response.ragGate),
      kind: response.ragGate.mode === 'prefetch' && response.ragGate.prefetched ? 'accent' : '',
      title: [response.ragGate.reason, ...(response.ragGate.hits || []).map((h) => `《${h.title}》 ${h.score}`)].join('\n')
    })
  }
  for (const skill of response.skillsApplied || []) {
    chips.push({
      text: `Skill · ${skill.id} ${Number(skill.score).toFixed(2)}`,
      kind: 'spark',
      title: (skill.reasons || []).join('；')
    })
  }
  for (const tool of response.toolsUsed || []) {
    if (tool !== 'search_knowledge_base') chips.push({ text: `工具 · ${tool}`, kind: '' })
  }
  return chips
}

function clearConversation() {
  messages.value = []
  lastResponse.value = null
  lastTrace.value = null
  settings.conversationId = ''
  persist()
}

async function searchKnowledge() {
  busy.value = true
  try {
    const data = await requestSearch(settings, searchQuery.value, 5, searchDomain.value)
    searchResults.value = data.results || []
    showToast(`检索完成，返回 ${searchResults.value.length} 条结果`)
  } catch (error) {
    statusText.value = error.message
    showToast('检索失败，请检查连接')
  } finally { busy.value = false }
}

async function submitKnowledge() {
  busy.value = true
  try {
    const data = await addKnowledge(settings, [{ title: docTitle.value.trim(), content: docContent.value.trim(), domain: docDomain.value }])
    statusText.value = JSON.stringify(data, null, 2)
    await loadStats()
    showToast('文档已添加')
  } catch (error) {
    statusText.value = error.message
    showToast('文档导入失败')
  } finally { busy.value = false }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busy.value = true
  try {
    const data = await uploadKnowledge(settings, file, docDomain.value)
    statusText.value = JSON.stringify(data, null, 2)
    await loadStats()
    showToast(`${file.name} 导入成功`)
  } catch (error) {
    statusText.value = error.message
    showToast('文件导入失败')
  } finally { busy.value = false }
}

async function runEvaluation() {
  busy.value = true
  try {
    evalData.value = await requestEvaluation(settings, compareRagGate.value ? { compare_rag_gate: true } : null)
    showToast('评测完成')
  } catch (error) {
    statusText.value = error.message
    showToast('评测运行失败')
  } finally { busy.value = false }
}

async function loadToolTrace(requestId) {
  try {
    return await requestToolTrace(settings, requestId)
  } catch {
    return null
  }
}

function formatPercent(value) {
  const number = Number(value || 0)
  return `${(number <= 1 ? number * 100 : number).toFixed(1)}%`
}

function formatJson(value) {
  try {
    return JSON.stringify(value ?? {}, null, 2)
  } catch {
    return String(value ?? '')
  }
}

function createMessageId() {
  messageSequence += 1
  return `message-${Date.now()}-${messageSequence}`
}

function showToast(message) {
  toast.value = message
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 2600)
}
</script>
