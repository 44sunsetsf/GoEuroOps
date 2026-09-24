// Sun position and sky colour for Stockholm, where the studio is based.
// Low-precision solar formulas (good to ~0.5°), plenty for painting a sky.

export const STOCKHOLM = { lat: 59.3293, lon: 18.0686, tz: 'Europe/Stockholm' }

const RAD = Math.PI / 180

export function sunPosition(date, { lat, lon } = STOCKHOLM) {
  const d = date.getTime() / 86400000 + 2440587.5 - 2451545.0
  const g = (357.529 + 0.98560028 * d) * RAD
  const q = 280.459 + 0.98564736 * d
  const L = (q + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * RAD
  const e = (23.439 - 0.00000036 * d) * RAD
  const ra = Math.atan2(Math.cos(e) * Math.sin(L), Math.cos(L))
  const dec = Math.asin(Math.sin(e) * Math.sin(L))
  const gmst = ((18.697374558 + 24.06570982441908 * d) % 24 + 24) % 24
  const h = (gmst * 15 + lon) * RAD - ra
  const phi = lat * RAD
  const elevation = Math.asin(Math.sin(phi) * Math.sin(dec) + Math.cos(phi) * Math.cos(dec) * Math.cos(h)) / RAD
  const azimuth = (Math.atan2(-Math.sin(h), Math.tan(dec) * Math.cos(phi) - Math.sin(phi) * Math.cos(h)) / RAD + 360) % 360
  return { elevation, azimuth }
}

// ── Stockholm wall-clock helpers ──────────────────────────────────────────
const partsFormatter = new Intl.DateTimeFormat('en-GB', {
  timeZone: STOCKHOLM.tz, hourCycle: 'h23',
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'
})

export function stockholmParts(date) {
  const p = Object.fromEntries(partsFormatter.formatToParts(date).map((x) => [x.type, x.value]))
  return { y: +p.year, m: +p.month, d: +p.day, minutes: +p.hour * 60 + +p.minute }
}

/** The instant when Stockholm clocks show y-m-d at `minutes` past midnight. */
export function stockholmDate(y, m, d, minutes) {
  const guess = Date.UTC(y, m - 1, d, 0, minutes)
  const shown = stockholmParts(new Date(guess))
  const offset = Date.UTC(shown.y, shown.m - 1, shown.d, 0, shown.minutes) - guess
  return new Date(guess - offset)
}

export function formatClock(minutes) {
  const m = ((Math.round(minutes) % 1440) + 1440) % 1440
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`
}

/** Sunrise, sunset and daylight for a Stockholm calendar day (minutes past local midnight). */
export function dayLight(y, m, d) {
  const up = (min) => sunPosition(stockholmDate(y, m, d, min)).elevation > -0.833
  let sunrise = null
  let sunset = null
  let prev = up(0)
  for (let min = 2; min <= 1440; min += 2) {
    const now = up(min)
    if (now && !prev && sunrise === null) sunrise = min
    if (!now && prev) sunset = min
    prev = now
  }
  const daylight = sunrise !== null && sunset !== null ? sunset - sunrise : null
  return { sunrise, sunset, daylight }
}

export function phaseOf(elevation) {
  if (elevation >= 6) return '白昼'
  if (elevation >= -4) return '金色时刻'
  if (elevation >= -8) return '蓝调时刻'
  return '夜晚'
}

// ── Sky palette, keyed by solar elevation ─────────────────────────────────
// [elevation, zenith, middle, horizon]
const SKY = [
  [-18, '#070d18', '#0d1829', '#16243a'],
  [-8, '#15243e', '#2b3e5f', '#4e6182'],
  [-4, '#2b4064', '#6c7fa2', '#c7a291'],
  [0, '#4a6488', '#c3988a', '#efb07a'],
  [6, '#6d91b3', '#d3c1ab', '#f3d19d'],
  [15, '#7ca5c5', '#bbd2de', '#ece5d0'],
  [35, '#6c9bc1', '#a8c7db', '#e1ebee']
]

function hex(c) {
  const n = parseInt(c.slice(1), 16)
  return [n >> 16, (n >> 8) & 255, n & 255]
}

export function mix(a, b, t) {
  const [x, y] = [hex(a), hex(b)]
  return '#' + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, '0')).join('')
}

export function skyColors(elevation) {
  const e = Math.max(SKY[0][0], Math.min(SKY[SKY.length - 1][0], elevation))
  let i = 0
  while (i < SKY.length - 2 && e > SKY[i + 1][0]) i++
  const [e0, ...c0] = SKY[i]
  const [e1, ...c1] = SKY[i + 1]
  const t = (e - e0) / (e1 - e0)
  const [zenith, middle, horizon] = c0.map((c, k) => mix(c, c1[k], t))
  return { zenith, middle, horizon }
}

/** 0 at deep night, 1 in full daylight: drives land tone and text contrast. */
export function lightLevel(elevation) {
  return Math.max(0, Math.min(1, (elevation + 10) / 22))
}
