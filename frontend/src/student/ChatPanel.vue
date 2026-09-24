<template>
  <Transition name="panel">
    <div v-if="open" class="chat-veil" @click.self="$emit('close')">
      <aside class="chat-panel" role="dialog" aria-modal="true" aria-labelledby="chat-title" @keydown.esc="$emit('close')">
        <header class="chat-head">
          <div>
            <h2 id="chat-title">问问指北</h2>
            <p>AI 助手先回答。涉及个人选校、文书和录取判断，会交给在瑞典的顾问。</p>
          </div>
          <button type="button" class="chat-close" aria-label="关闭对话" @click="$emit('close')">
            <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>
          </button>
        </header>

        <div ref="list" class="chat-list">
          <div v-if="!messages.length" class="chat-welcome">
            <p>你好，我是指北的 AI 助手。可以问我五国的申请时间、语言要求、学费，也可以了解我们的服务和价格。</p>
            <div class="chat-starters">
              <button v-for="q in STARTERS" :key="q" type="button" @click="send(q)">{{ q }}</button>
            </div>
          </div>

          <article v-for="m in messages" :key="m.id" :class="['bubble', m.role]">
            <div v-if="m.role === 'assistant' && m.pending && !m.content" class="bubble-wait" aria-label="正在回答">
              <span></span><span></span><span></span>
            </div>
            <div v-else-if="m.role === 'assistant'" class="bubble-md" v-html="renderMarkdown(m.content)"></div>
            <p v-else>{{ m.content }}</p>
            <p v-if="m.handoff" class="bubble-note">已转给顾问，工作日 24 小时内会联系你。</p>
            <p v-if="m.failed" class="bubble-note is-error">{{ m.failed }}</p>
          </article>
        </div>

        <form class="chat-compose" @submit.prevent="send()">
          <label class="visually-hidden" for="chat-input">输入你的问题</label>
          <textarea
            id="chat-input"
            ref="input"
            v-model="draft"
            rows="1"
            placeholder="比如：我想申请 KTH，需要准备什么？"
            @keydown.enter.exact.prevent="send()"
            @input="grow"
          ></textarea>
          <button type="submit" :disabled="busy || !draft.trim()">发送</button>
        </form>
      </aside>
    </div>
  </Transition>
</template>

<script setup>
import { nextTick, reactive, ref, watch } from 'vue'
import { streamChat } from '../lib/backends'
import { renderMarkdown } from '../lib/markdown'
import { STARTERS } from './content'

const props = defineProps({ open: Boolean, prompt: { type: String, default: '' } })
defineEmits(['close'])

const KEY = 'goeuroops.student.chat'
const settings = reactive(loadSettings())
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const list = ref(null)
const input = ref(null)
let seq = 0

watch(() => props.open, async (isOpen) => {
  if (!isOpen) return
  if (props.prompt) draft.value = props.prompt
  await nextTick()
  input.value?.focus()
  grow()
})

function loadSettings() {
  let saved = {}
  try { saved = JSON.parse(localStorage.getItem(KEY) || '{}') } catch { /* private mode */ }
  return {
    userId: saved.userId || `guest-${Math.random().toString(36).slice(2, 8)}`,
    conversationId: saved.conversationId || '',
    apiUrl: ''
  }
}

function persist() {
  try { localStorage.setItem(KEY, JSON.stringify({ userId: settings.userId, conversationId: settings.conversationId })) } catch { /* ignore */ }
}

function failureText(error) {
  const message = String(error?.message || '')
  if (message.includes('名额')) return '今天的体验名额已经用完了，明天再来看看吧。'
  if (message.startsWith('429')) return '发得有点快了，歇一分钟再问吧。'
  return '没有连上指北的服务。检查网络后再发一次试试。'
}

function grow() {
  const el = input.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 160)}px`
}

function scrollDown() {
  nextTick(() => list.value?.scrollTo({ top: list.value.scrollHeight, behavior: 'smooth' }))
}

async function send(text) {
  const content = (text ?? draft.value).trim()
  if (!content || busy.value) return
  draft.value = ''
  nextTick(grow)
  messages.value.push({ id: ++seq, role: 'user', content })
  const reply = reactive({ id: ++seq, role: 'assistant', content: '', pending: true, handoff: false, failed: '' })
  messages.value.push(reply)
  busy.value = true
  scrollDown()
  try {
    await streamChat(settings, content, {
      onToken(t) { reply.content += t; scrollDown() },
      onDone(res) {
        if (!reply.content) reply.content = res.response
        if (res.conversationId) { settings.conversationId = res.conversationId; persist() }
        reply.handoff = Boolean(res.escalated)
      },
      onError(error) { reply.failed = failureText(error) }
    })
  } catch (error) {
    reply.failed = failureText(error)
  } finally {
    reply.pending = false
    busy.value = false
    scrollDown()
  }
}
</script>
