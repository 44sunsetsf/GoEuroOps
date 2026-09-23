<template>
  <section class="page page-scroll page-skills">
    <div class="page-heading">
      <div class="heading-copy">
        <h1>Skills</h1>
        <p>
          每个 Skill 是一份业务规范（<code>skills/&lt;id&gt;/SKILL.md</code>）。命中后注入对应 Agent 的 system prompt，
          references 里的详细资料由模型按需读取。改完文件点「重新加载」即可生效，不用重启。
        </p>
      </div>
      <div class="heading-actions">
        <button class="quiet-button" @click="runEvals" :disabled="busy">运行命中回归</button>
        <button @click="reload" :disabled="busy">重新加载</button>
      </div>
    </div>

    <div v-if="data.errors?.length" class="inline-error">
      <strong>{{ data.errors.length }} 个 Skill 加载失败：</strong>
      <div v-for="err in data.errors" :key="err" class="mono small">{{ err }}</div>
    </div>

    <div v-if="evals" :class="['eval-banner', evals.accuracy >= 0.9 ? 'ok' : 'warn']">
      <strong>命中回归 {{ evals.passed }}/{{ evals.total }}（{{ (evals.accuracy * 100).toFixed(0) }}%）</strong>
      <span v-if="!evals.failures.length">全部通过</span>
      <ul v-else>
        <li v-for="f in evals.failures" :key="f.skill + f.message">
          {{ f.skill }}：「{{ f.message }}」期望{{ f.expect_hit === false ? '不命中' : '命中' }}，实际{{ f.hit ? '命中' : '未命中' }}
        </li>
      </ul>
    </div>

    <div class="skills-layout">
      <div class="skill-cards">
        <article v-for="skill in data.skills" :key="skill.id" class="skill-card">
          <header>
            <div>
              <strong>{{ skill.name }}</strong>
              <code>{{ skill.id }} · v{{ skill.version }}</code>
            </div>
            <span :class="['chip', skill.mode === 'always' ? 'chip-spark' : 'chip-accent']">{{ skill.mode === 'always' ? '常驻' : '按需' }}</span>
          </header>
          <p class="skill-desc">{{ skill.description }}</p>
          <dl class="skill-meta">
            <div><dt>Agent</dt><dd>{{ skill.agents.length ? skill.agents.map((a) => label(AGENT_LABELS, a)).join('、') : '全部' }}</dd></div>
            <div v-if="skill.intents.length"><dt>绑定意图</dt><dd>{{ skill.intents.map((i) => label(INTENT_LABELS, i)).join('、') }}</dd></div>
            <div><dt>触发词 / 样例</dt><dd>{{ skill.keywords.length }} / {{ skill.examples.length }}</dd></div>
            <div><dt>正文</dt><dd>{{ skill.content_chars }} 字 · #{{ skill.content_hash }}</dd></div>
            <div><dt>维护</dt><dd>{{ skill.owner || '-' }} · {{ skill.updated_at || '-' }}</dd></div>
            <div><dt>命中</dt><dd><b>{{ skill.stats?.hits ?? 0 }}</b> 次<span v-if="skill.stats?.last_hit_at"> · 最近 {{ formatTime(skill.stats.last_hit_at) }}</span></dd></div>
          </dl>
          <div v-if="skill.references.length" class="ref-chips">
            <span v-for="ref in skill.references" :key="ref.file" class="chip" :title="ref.title">{{ ref.file }}</span>
          </div>
          <p v-for="w in skill.warnings" :key="w" class="skill-warning">{{ w }}</p>
          <button class="link-button tiny" @click="toggleDetail(skill.id)">{{ openId === skill.id ? '收起正文' : '查看正文' }}</button>
          <pre v-if="openId === skill.id" class="skill-body">{{ detail?.content || '加载中…' }}</pre>
        </article>
        <div v-if="!data.skills.length" class="workspace-empty">暂无已加载 Skill。</div>
      </div>

      <aside class="workspace-card match-card">
        <div class="card-heading"><div><h2>命中测试</h2></div><code>POST /skills/match</code></div>
        <p class="hint">输入一句用户可能说的话，看每个 Skill 的得分。不指定意图时会现场做一次意图识别（与线上链路一致）。</p>
        <textarea v-model="probe.message" rows="3" placeholder="例如：想去北欧读计算机研究生"></textarea>
        <div class="probe-options">
          <label>
            <span>Agent</span>
            <select v-model="probe.agent">
              <option value="">不限</option>
              <option v-for="(text, key) in AGENT_LABELS" :key="key" :value="key">{{ text }}</option>
            </select>
          </label>
          <label>
            <span>意图</span>
            <select v-model="probe.intent">
              <option value="">自动识别</option>
              <option v-for="(text, key) in INTENT_LABELS" :key="key" :value="key">{{ text }}</option>
            </select>
          </label>
        </div>
        <button @click="runMatch" :disabled="busy || !probe.message.trim()">测试</button>

        <div v-if="match" class="match-result">
          <p class="match-summary">
            意图 <b>{{ label(INTENT_LABELS, match.intent) }}</b>
            <span v-if="match.intent_confidence != null">（{{ (match.intent_confidence * 100).toFixed(0) }}%）</span>
            · 阈值 {{ match.threshold }} · 注入 {{ match.injected.length }} 个 / {{ match.prompt_chars }} 字
          </p>
          <div v-for="row in match.results" :key="row.id" :class="['match-row', row.status]">
            <div class="match-row-head">
              <strong>{{ row.name }}</strong>
              <span>{{ STATUS_TEXT[row.status] }}</span>
            </div>
            <div class="score-bar">
              <i :style="{ width: `${Math.min(row.score, 1) * 100}%` }"></i>
              <b :style="{ left: `${match.threshold * 100}%` }"></b>
            </div>
            <div class="signal-chips">
              <span class="mono">{{ row.score.toFixed(2) }}</span>
              <span v-for="(value, key) in row.signals" :key="key" class="chip">{{ SIGNAL_TEXT[key] || key }} +{{ value.toFixed(2) }}</span>
            </div>
            <p v-if="row.reasons.length" class="match-reason">{{ row.reasons.join('；') }}</p>
          </div>
        </div>
      </aside>
    </div>
  </section>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { matchSkills, reloadSkills, requestSkillDetail, requestSkills, runSkillEvals } from '../lib/backends'
import { AGENT_LABELS, INTENT_LABELS, formatTime, label } from '../lib/labels'

const props = defineProps({ settings: { type: Object, required: true } })
const emit = defineEmits(['toast'])

const STATUS_TEXT = {
  matched: '会注入',
  below_threshold: '未达阈值',
  agent_filtered: '不适用该 Agent',
  disabled: '已停用'
}
const SIGNAL_TEXT = { always: '常驻', intent: '意图', keywords: '关键词', semantic: '语义', sticky: '会话' }

const data = ref({ skills: [], errors: [] })
const busy = ref(false)
const evals = ref(null)
const match = ref(null)
const openId = ref('')
const detail = ref(null)
const probe = reactive({ message: '想去北欧读计算机研究生', agent: 'consulting', intent: '' })

onMounted(load)

async function load() {
  try {
    data.value = await requestSkills(props.settings)
  } catch (err) {
    emit('toast', `Skills 加载失败：${err.message}`)
  }
}

async function reload() {
  busy.value = true
  try {
    data.value = await reloadSkills(props.settings)
    emit('toast', `已重新加载 ${data.value.count} 个 Skill`)
  } catch (err) {
    emit('toast', `重新加载失败：${err.message}`)
  } finally { busy.value = false }
}

async function runEvals() {
  busy.value = true
  try {
    evals.value = await runSkillEvals(props.settings)
  } catch (err) {
    emit('toast', `回归运行失败：${err.message}`)
  } finally { busy.value = false }
}

async function runMatch() {
  busy.value = true
  try {
    match.value = await matchSkills(props.settings, {
      message: probe.message,
      agent_type: probe.agent || null,
      intent: probe.intent || null
    })
  } catch (err) {
    emit('toast', `测试失败：${err.message}`)
  } finally { busy.value = false }
}

async function toggleDetail(id) {
  if (openId.value === id) { openId.value = ''; return }
  openId.value = id
  detail.value = null
  try {
    detail.value = await requestSkillDetail(props.settings, id)
  } catch (err) {
    detail.value = { content: err.message }
  }
}
</script>
