<template>
  <section class="page page-scroll page-leads">
    <div class="page-heading">
      <div class="heading-copy">
        <h1>线索</h1>
        <p>助手在对话里登记的咨询线索（用户同意后才会登记）和转顾问交接单。顾问在这里跟进状态、写备注。</p>
      </div>
      <div class="heading-actions">
        <button class="quiet-button" @click="load" :disabled="loading">{{ loading ? '加载中' : '刷新' }}</button>
      </div>
    </div>

    <div class="stat-strip">
      <button
        v-for="item in statusTabs"
        :key="item.value"
        :class="['stat-tile', { active: filters.status === item.value }]"
        @click="setStatus(item.value)"
      >
        <strong>{{ item.count }}</strong>
        <span>{{ item.label }}</span>
      </button>
      <div class="stat-tile stat-tile-static">
        <strong>{{ stats.by_type?.handoff ?? 0 }}</strong>
        <span>转顾问单</span>
      </div>
    </div>

    <div class="toolbar-line">
      <div class="segmented">
        <button :class="{ active: filters.type === '' }" @click="setType('')">全部</button>
        <button :class="{ active: filters.type === 'lead' }" @click="setType('lead')">咨询线索</button>
        <button :class="{ active: filters.type === 'handoff' }" @click="setType('handoff')">转顾问</button>
      </div>
      <span class="toolbar-note">存储：{{ stats.backend === 'redis' ? 'Redis（持久化）' : stats.backend === 'memory' ? '内存（Redis 不可用，重启会丢失）' : '-' }}</span>
    </div>

    <div v-if="error" class="inline-error">
      {{ error }}
      <span v-if="error.includes('401')">请在「对话」页右侧的连接配置里填写 Admin Token。</span>
    </div>

    <div v-if="leads.length" class="lead-list">
      <article v-for="lead in leads" :key="lead.id" :class="['lead-card', `lead-${lead.status}`]">
        <header class="lead-head">
          <div class="lead-title">
            <span :class="['type-badge', lead.type]">{{ label(LEAD_TYPE_LABELS, lead.type) }}</span>
            <strong>{{ lead.name || lead.user_id || '未留称呼' }}</strong>
            <code>{{ lead.id }}</code>
          </div>
          <span class="lead-time">{{ formatTime(lead.created_at) }}</span>
        </header>

        <dl class="lead-fields">
          <div v-if="lead.contact">
            <dt>{{ label(CHANNEL_LABELS, lead.contact_channel) }}</dt>
            <dd>
              <span class="mono">{{ revealed[lead.id] ? lead.contact : mask(lead.contact) }}</span>
              <button class="link-button tiny" @click="toggleReveal(lead.id)">{{ revealed[lead.id] ? '隐藏' : '显示' }}</button>
            </dd>
          </div>
          <div v-if="lead.countries?.length"><dt>目标国家</dt><dd>{{ lead.countries.join('、') }}</dd></div>
          <div v-if="lead.stage"><dt>阶段</dt><dd>{{ label(STAGE_LABELS, lead.stage) }}</dd></div>
          <div v-if="lead.target_intake"><dt>入学</dt><dd>{{ lead.target_intake }}</dd></div>
          <div v-if="lead.interested_services?.length"><dt>意向服务</dt><dd>{{ lead.interested_services.join('、') }}</dd></div>
          <div v-if="lead.preferred_time"><dt>方便时间</dt><dd>{{ lead.preferred_time }}</dd></div>
          <div v-if="lead.reason"><dt>转交原因</dt><dd>{{ lead.reason }}</dd></div>
          <div v-if="lead.intent"><dt>意图</dt><dd>{{ label(INTENT_LABELS, lead.intent) }}</dd></div>
        </dl>
        <p v-if="lead.background" class="lead-quote">{{ lead.background }}</p>
        <p v-if="lead.last_message" class="lead-quote">“{{ lead.last_message }}”</p>

        <footer class="lead-actions">
          <select :value="lead.status" @change="save(lead, { status: $event.target.value })">
            <option v-for="item in LEAD_STATUS" :key="item.value" :value="item.value">{{ item.label }}</option>
          </select>
          <input
            v-model="notes[lead.id]"
            placeholder="跟进备注，例如：已加微信，约周四 20:00（北京时间）"
            @keydown.enter="save(lead, { notes: notes[lead.id] || '' })"
          />
          <button class="quiet-button" @click="save(lead, { notes: notes[lead.id] || '' })">保存备注</button>
        </footer>
      </article>
    </div>
    <div v-else-if="!loading && !error" class="evaluation-empty">
      <div class="empty-symbol">◇</div>
      <h2>暂时没有线索</h2>
      <p>在对话里说"我想预约，微信是 xxx"，助手征得同意后会登记到这里。</p>
    </div>
  </section>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { requestLeads, updateLead } from '../lib/backends'
import {
  CHANNEL_LABELS,
  INTENT_LABELS,
  LEAD_STATUS,
  LEAD_TYPE_LABELS,
  STAGE_LABELS,
  formatTime,
  label
} from '../lib/labels'

const props = defineProps({ settings: { type: Object, required: true } })
const emit = defineEmits(['toast'])

const leads = ref([])
const stats = ref({ total: 0, by_status: {}, by_type: {} })
const filters = reactive({ status: '', type: '' })
const notes = reactive({})
const revealed = reactive({})
const loading = ref(false)
const error = ref('')

const statusTabs = computed(() => [
  { value: '', label: '全部', count: stats.value.total ?? 0 },
  ...LEAD_STATUS.map((item) => ({ ...item, count: stats.value.by_status?.[item.value] ?? 0 }))
])

onMounted(load)

async function load() {
  loading.value = true
  error.value = ''
  try {
    const data = await requestLeads(props.settings, filters)
    leads.value = data.items || []
    stats.value = data.stats || stats.value
    for (const lead of leads.value) {
      if (!(lead.id in notes)) notes[lead.id] = lead.notes || ''
    }
  } catch (err) {
    error.value = err.message
  } finally {
    loading.value = false
  }
}

function setStatus(value) { filters.status = value; load() }
function setType(value) { filters.type = value; load() }
function toggleReveal(id) { revealed[id] = !revealed[id] }

function mask(value) {
  const text = String(value || '')
  if (text.includes('@')) {
    const [name, domain] = text.split('@')
    return `${name.slice(0, 2)}***@${domain}`
  }
  if (text.length <= 4) return '*'.repeat(text.length)
  return `${text.slice(0, 2)}${'*'.repeat(text.length - 4)}${text.slice(-2)}`
}

async function save(lead, patch) {
  try {
    const updated = await updateLead(props.settings, lead.id, patch)
    Object.assign(lead, updated)
    emit('toast', '已保存')
    if (patch.status) load()
  } catch (err) {
    emit('toast', `保存失败：${err.message}`)
  }
}
</script>
