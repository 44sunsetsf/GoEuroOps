<template>
  <section class="sky" :class="tone" :style="{ '--sky-horizon': colors.horizon }">
    <svg class="sky-art" viewBox="0 0 1440 800" :preserveAspectRatio="narrow ? 'xMinYMax slice' : 'xMidYMax slice'" aria-hidden="true">
      <defs>
        <linearGradient id="sky-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" :stop-color="colors.zenith" />
          <stop offset="0.62" :stop-color="colors.middle" />
          <stop offset="1" :stop-color="colors.horizon" />
        </linearGradient>
        <linearGradient id="water-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" :stop-color="mix(colors.horizon, land.base, 0.18)" />
          <stop offset="0.5" :stop-color="mix(colors.middle, land.base, 0.35)" />
          <stop offset="1" :stop-color="mix(colors.zenith, land.base, 0.45)" />
        </linearGradient>
        <radialGradient id="sun-glow">
          <stop offset="0" :stop-color="sunColor" stop-opacity="0.55" />
          <stop offset="1" :stop-color="sunColor" stop-opacity="0" />
        </radialGradient>
        <radialGradient id="window-glow">
          <stop offset="0" stop-color="#f6c66b" stop-opacity="0.8" />
          <stop offset="1" stop-color="#f6c66b" stop-opacity="0" />
        </radialGradient>
        <clipPath id="above-horizon"><rect width="1440" height="560" /></clipPath>
      </defs>

      <rect width="1440" height="560" fill="url(#sky-grad)" />

      <g :opacity="starOpacity">
        <circle v-for="(s, i) in STARS" :key="i" :cx="s[0]" :cy="s[1]" :r="s[2]" fill="#f4f1e6" />
      </g>

      <g clip-path="url(#above-horizon)">
        <circle :cx="sunX" :cy="sunY" r="150" fill="url(#sun-glow)" />
        <circle :cx="sunX" :cy="sunY" r="24" :fill="sunColor" />
      </g>

      <rect y="560" width="1440" height="240" fill="url(#water-grad)" />
      <g v-if="sunUp" :opacity="Math.min(1, 0.25 + (30 - Math.min(30, position.elevation)) / 40)">
        <rect v-for="(r, i) in GLINTS" :key="i" :x="sunX - r[1] / 2 + r[2]" :y="566 + r[0]" :width="r[1]" height="2" rx="1" :fill="sunColor" :opacity="0.7 - i * 0.07" />
      </g>
      <g stroke="#ffffff" stroke-width="1" :opacity="0.08 + light * 0.1">
        <line x1="120" y1="640" x2="330" y2="640" />
        <line x1="930" y1="662" x2="1180" y2="662" />
        <line x1="560" y1="700" x2="700" y2="700" />
        <line x1="1220" y1="728" x2="1380" y2="728" />
      </g>

      <!-- far skerries -->
      <path d="M0 560 C110 541 250 536 360 551 L392 560 Z" :fill="land.far" />
      <path d="M980 560 C1080 543 1190 539 1300 549 C1370 555 1420 557 1440 558 L1440 560 Z" :fill="land.far" />
      <!-- the pine island -->
      <path d="M700 560 C770 526 890 512 1000 523 C1052 529 1094 545 1124 560 Z" :fill="land.mid" />
      <g :fill="land.mid">
        <path v-for="(p, i) in PINES" :key="i" :d="`M${p[0]} ${p[1]} l${p[2] / 2} ${-p[3]} l${p[2] / 2} ${p[3]} Z`" />
      </g>
      <path d="M700 560 C770 594 890 606 1000 598 C1052 594 1094 575 1124 560 Z" :fill="land.mid" opacity="0.18" />

      <!-- near granite with the red stuga -->
      <path d="M-20 612 C60 572 190 556 330 562 C410 566 470 584 520 612 Z" :fill="land.near" />
      <g transform="translate(214 522)">
        <rect x="0" y="16" width="58" height="30" :fill="cabinWall" />
        <path d="M-5 17 L29 -6 L63 17 Z" :fill="land.near" />
        <rect x="0" y="16" width="2.5" height="30" fill="#eef0ec" :opacity="0.4 + light * 0.5" />
        <rect x="55.5" y="16" width="2.5" height="30" fill="#eef0ec" :opacity="0.4 + light * 0.5" />
        <circle v-if="windowLit" cx="19" cy="28" r="28" fill="url(#window-glow)" />
        <rect x="13" y="23" width="12" height="10" :fill="windowLit ? '#f6c66b' : mix('#cfd8dc', land.near, 0.4)" />
        <rect x="36" y="25" width="10" height="21" :fill="mix('#eef0ec', land.near, 0.55)" />
      </g>
      <path d="M-20 612 C60 646 190 660 330 656 C410 652 470 636 520 612 Z" :fill="land.near" opacity="0.22" />
    </svg>

    <div class="sky-content">
      <slot />
    </div>

    <div class="sky-clock" aria-live="polite">
      <div class="clock-read">
        <strong>斯德哥尔摩 {{ clock }}</strong>
        <span>{{ phase }}</span>
      </div>
      <p class="clock-note">{{ caption }}</p>
      <div class="clock-controls">
        <div class="clock-modes" role="group" aria-label="选择日子">
          <button v-for="m in MODES" :key="m.id" type="button" :aria-pressed="mode === m.id" @click="setMode(m.id)">{{ m.label }}</button>
        </div>
        <label class="clock-slider">
          <span class="visually-hidden">拖动查看一天里的光线</span>
          <input type="range" min="0" max="1439" step="1" :value="Math.round(minutes)" @input="scrub(+$event.target.value)" />
        </label>
        <button v-if="mode === 'today' && frozen" type="button" class="clock-reset" @click="resume">回到此刻</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { dayLight, formatClock, lightLevel, mix, phaseOf, skyColors, stockholmDate, stockholmParts, sunPosition } from './sun'

const MODES = [
  { id: 'today', label: '今天' },
  { id: 'midsummer', label: '夏至' },
  { id: 'midwinter', label: '冬至' }
]

const STARS = Array.from({ length: 70 }, (_, i) => {
  const r = (n) => ((Math.sin(i * 127.1 + n * 311.7) * 43758.5453) % 1 + 1) % 1
  return [r(1) * 1440, r(2) * 470, 0.5 + r(3) * 1.1]
})
const PINES = [[800, 530, 12, 26], [818, 526, 14, 34], [838, 522, 11, 24], [862, 518, 15, 38], [884, 516, 12, 28],
  [905, 515, 14, 33], [930, 516, 11, 22], [950, 518, 13, 30], [972, 521, 10, 20], [990, 524, 12, 25]]
const GLINTS = [[0, 90, 0], [9, 64, 6], [18, 46, -4], [28, 30, 5], [40, 18, -2], [54, 10, 3]]

const today = () => stockholmParts(new Date())
const mode = ref('today')
// On phones, anchor the scene left so the red stuga stays in frame.
const narrow = ref(window.matchMedia?.('(max-width: 700px)').matches ?? false)
const frozen = ref(false)
const minutes = ref(today().minutes)
let timer
let tween

const dayOf = computed(() => {
  const t = today()
  if (mode.value === 'today') return [t.y, t.m, t.d]
  const [m, d] = mode.value === 'midsummer' ? [6, 21] : [12, 21]
  const passed = t.m > m || (t.m === m && t.d > d)
  return [passed ? t.y + 1 : t.y, m, d]
})

const position = computed(() => sunPosition(stockholmDate(...dayOf.value, minutes.value)))
const colors = computed(() => skyColors(position.value.elevation))
const light = computed(() => lightLevel(position.value.elevation))
const tone = computed(() => (light.value > 0.62 ? 'tone-day' : 'tone-night'))
const sun = computed(() => dayLight(...dayOf.value))

const land = computed(() => {
  const base = mix('#0a121b', '#34505d', light.value)
  return {
    base,
    far: mix(base, colors.value.horizon, 0.5),
    mid: mix(base, colors.value.horizon, 0.22),
    near: mix(base, '#05090e', 0.25)
  }
})
const cabinWall = computed(() => mix('#8e2f25', '#1c0f0d', 0.6 - light.value * 0.6))
const windowLit = computed(() => position.value.elevation < -1)
const sunUp = computed(() => position.value.elevation > -2)
const sunColor = computed(() => mix('#f4a45f', '#fff3d8', Math.max(0, Math.min(1, position.value.elevation / 25))))
const sunX = computed(() => 720 + ((position.value.azimuth - 180) / 135) * 720)
const sunY = computed(() => 560 - position.value.elevation * 13)
const starOpacity = computed(() => Math.max(0, Math.min(0.9, (-position.value.elevation - 7) / 8)))

const clock = computed(() => formatClock(minutes.value))
const phase = computed(() => phaseOf(position.value.elevation))
const caption = computed(() => {
  const { sunrise, sunset, daylight } = sun.value
  const hours = daylight ? `${Math.floor(daylight / 60)} 小时 ${daylight % 60} 分` : ''
  if (mode.value === 'midsummer') {
    return `夏至这天，太阳 ${formatClock(sunset)} 才落下，${formatClock(sunrise)} 又升起来。午夜的天空只是一层淡蓝。`
  }
  if (mode.value === 'midwinter') {
    return `冬至这天，日照只有 ${hours}，下午 ${Math.floor(sunset / 60) - 12} 点多天就黑了。瑞典人用蜡烛和 fika 过冬。`
  }
  return `今天日出 ${formatClock(sunrise)}，日落 ${formatClock(sunset)}，日照 ${hours}。`
})

function animateTo(target) {
  cancelAnimationFrame(tween)
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  if (reduce) { minutes.value = target; return }
  const from = minutes.value
  const start = performance.now()
  const step = (now) => {
    const t = Math.min(1, (now - start) / 900)
    const eased = 1 - Math.pow(1 - t, 3)
    minutes.value = from + (target - from) * eased
    if (t < 1) tween = requestAnimationFrame(step)
  }
  tween = requestAnimationFrame(step)
}

function setMode(id) {
  mode.value = id
  if (id === 'today') { resume(); return }
  frozen.value = true
  // Land on the most telling moment of each day.
  animateTo(id === 'midsummer' ? 23 * 60 + 30 : 15 * 60 + 30)
}

function scrub(value) {
  cancelAnimationFrame(tween)
  frozen.value = true
  minutes.value = value
}

function resume() {
  frozen.value = false
  mode.value = 'today'
  animateTo(today().minutes)
}

onMounted(() => {
  timer = setInterval(() => {
    if (mode.value === 'today' && !frozen.value) minutes.value = today().minutes
  }, 30000)
})
onBeforeUnmount(() => {
  clearInterval(timer)
  cancelAnimationFrame(tween)
})
</script>
