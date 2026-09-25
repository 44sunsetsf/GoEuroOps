<template>
  <div class="student">
    <header class="nav" :class="{ solid: scrolled }">
      <a class="nav-brand" href="/" aria-label="指北首页">
        <svg class="compass" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="10.5" fill="none" stroke="currentColor" stroke-width="1.2" />
          <path d="M12 3.5 L14.6 12 L12 11 L9.4 12 Z" fill="#8e2f25" />
          <path d="M12 20.5 L9.4 12 L12 13 L14.6 12 Z" fill="currentColor" opacity="0.45" />
        </svg>
        <span>指北</span>
      </a>
      <nav class="nav-links" aria-label="页面导航">
        <a href="#countries">五个国家</a>
        <a href="#menu">服务与价格</a>
        <a href="#us">我们是谁</a>
      </nav>
      <button type="button" class="nav-ask" @click="ask()">问问指北</button>
    </header>

    <NordicSky>
      <h1 class="hero-title">去北欧名校，<br />读计算机硕士</h1>
      <p class="hero-lede">
        指北由几位在瑞典读 CS 的学长学姐创办，只做瑞典、芬兰、丹麦、荷兰和德国的英语授课计算机硕士申请。
        这条路我们自己走过，愿意陪你再走一遍。
      </p>
      <div class="hero-actions">
        <button type="button" class="btn-primary" @click="ask()">问问指北</button>
        <button type="button" class="btn-ghost" @click="ask('我想预约一次 15 分钟的免费初步沟通')">预约 15 分钟免费沟通</button>
      </div>
    </NordicSky>

    <section id="countries" class="band band-countries">
      <div class="wrap">
        <div class="band-head">
          <h2>五个国家，从北往南</h2>
          <p>按首都的纬度排列。每个国家的申请节奏、学费和生活都不一样，选之前值得先弄清楚。</p>
        </div>
        <ol class="latitudes">
          <li v-for="c in COUNTRIES" :key="c.en" class="latitude">
            <div class="lat-mark">
              <span class="lat-num">{{ c.lat.toFixed(1) }}°N</span>
              <span class="lat-city">{{ c.city }}</span>
            </div>
            <div class="lat-body">
              <h3>{{ c.name }} <span lang="en">{{ c.en }}</span></h3>
              <p>{{ c.line }}</p>
              <p class="lat-schools">常见的选择：{{ c.schools }}</p>
            </div>
            <button type="button" class="lat-ask" @click="ask(c.ask)">问问{{ c.name }}的申请</button>
          </li>
        </ol>
        <p class="fineprint">以上是通用参考，具体要求每年会变，以各校官网为准。</p>
      </div>
    </section>

    <section id="menu" class="band band-menu">
      <div class="wrap menu-wrap">
        <div class="band-head">
          <h2>服务与价格</h2>
          <p>价格公开，不议价。每项服务都签电子协议，写清内容、修改轮次和交付时间。</p>
          <p v-if="earlyBird" class="early-bird">{{ earlyBird }}</p>
        </div>
        <ul class="menu">
          <li v-for="item in menu" :key="item.sku">
            <div class="menu-line">
              <h3>{{ item.name }}</h3>
              <span class="menu-leader" aria-hidden="true"></span>
              <strong>{{ formatPrice(item.price) }}</strong>
            </div>
            <p>{{ item.summary }} <span class="menu-unit">{{ item.unit }}</span></p>
          </li>
        </ul>
      </div>
    </section>

    <section id="us" class="band band-us">
      <div class="wrap us-wrap">
        <div class="us-letter">
          <h2>我们是谁</h2>
          <p>
            我们是几个在瑞典读计算机的中国学生。申请的时候，我们也在深夜对着 universityadmissions.se 发愁，
            也不知道动机信该从哪里写起。后来到了这边，才知道北欧的课堂、冬天和 fika 是什么样子。
          </p>
          <p>
            所以指北只做自己真正懂的事：五个国家的英语授课 CS 硕士。选校和文书由我们亲自完成，每个申请季只带很少的学员。
          </p>
          <p class="us-sign">指北，写于斯德哥尔摩</p>
        </div>
        <dl class="promises">
          <div v-for="p in PROMISES" :key="p.title">
            <dt>{{ p.title }}</dt>
            <dd>{{ p.body }}</dd>
          </div>
        </dl>
      </div>
    </section>

    <section class="band band-close">
      <div class="wrap close-wrap">
        <h2>有问题，先问一句。</h2>
        <p>AI 助手随时在，顾问工作日 24 小时内回复。第一次沟通 15 分钟，免费。</p>
        <button type="button" class="btn-primary" @click="ask()">问问指北</button>
      </div>
    </section>

    <footer class="foot">
      <div class="wrap foot-wrap">
        <span>指北 Nordic CS Master Studio，斯德哥尔摩</span>
        <span>价格与政策以服务协议为准</span>
        <a href="/studio">工作室后台</a>
      </div>
    </footer>

    <!-- 作品说明：这里只是学生端，作品的主体是工作室后台（智能运营中枢），入口要让人看得见 -->
    <a class="studio-badge" href="/studio">
      <svg viewBox="0 0 24 24" aria-hidden="true"><rect x="5" y="10.5" width="14" height="10" rx="2" /><path d="M8.5 10.5V7.5a3.5 3.5 0 0 1 7 0v3" /></svg>
      <span class="studio-badge-text">
        <b>智能运营中枢</b>
        <small>这个作品的主体在后台：线索、价目、Skills、知识库、评测 · 需口令</small>
      </span>
      <span class="studio-badge-go" aria-hidden="true">→</span>
    </a>

    <ChatPanel :open="chatOpen" :prompt="chatPrompt" @close="chatOpen = false" />
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { requestCatalog } from '../lib/backends'
import { stockholmParts } from './sun'
import { COUNTRIES, MENU, PROMISES } from './content'
import NordicSky from './NordicSky.vue'
import ChatPanel from './ChatPanel.vue'
import './student.css'

const chatOpen = ref(false)
const chatPrompt = ref('')
const scrolled = ref(false)
const catalog = ref(null)

const menu = computed(() => {
  const live = Object.fromEntries((catalog.value?.services || []).map((s) => [s.sku, s]))
  return MENU.map((item) => ({ ...item, price: live[item.sku]?.price ?? item.price }))
})

const earlyBird = computed(() => {
  const rule = catalog.value?.discounts?.early_bird
  if (!rule?.deadline_month_day) return ''
  const [m, d] = rule.deadline_month_day.split('-').map(Number)
  const t = stockholmParts(new Date())
  if (t.m > m || (t.m === m && t.d > d)) return ''
  return `${m} 月 ${d} 日前签约，选校全案、文书套餐和全程陪跑享 ${Math.round(rule.rate * 100) / 10} 折。`
})

function formatPrice(price) {
  return price === 0 ? '免费' : `¥${Number(price).toLocaleString('en-US')}`
}

function ask(prompt = '') {
  chatPrompt.value = prompt
  chatOpen.value = true
}

function onScroll() { scrolled.value = window.scrollY > 40 }

onMounted(async () => {
  document.title = '指北 · 北欧 CS 硕士申请'
  window.addEventListener('scroll', onScroll, { passive: true })
  try {
    catalog.value = (await requestCatalog({ apiUrl: '' })).catalog
  } catch { /* the static menu is already on screen */ }
})
onBeforeUnmount(() => window.removeEventListener('scroll', onScroll))
</script>
