// Two-language UI. The Chinese source string is the key: t('中文') returns it unchanged in
// Chinese and looks it up in en.js otherwise, so a missing translation shows Chinese, never a blank.
// Placeholders are written {name} in the key and filled from the second argument.
import { ref } from 'vue'
import EN from './en.js'

const STORAGE_KEY = 'goeuroops.lang'

function initialLang() {
  try {
    const fromUrl = new URLSearchParams(window.location.search).get('lang')
    if (fromUrl === 'zh' || fromUrl === 'en') {
      localStorage.setItem(STORAGE_KEY, fromUrl)
      return fromUrl
    }
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved === 'zh' || saved === 'en') return saved
  } catch { /* private mode: fall through to the browser language */ }
  return (navigator.language || '').toLowerCase().startsWith('zh') ? 'zh' : 'en'
}

export const lang = ref(initialLang())

export function setLang(value) {
  lang.value = value
  try { localStorage.setItem(STORAGE_KEY, value) } catch { /* ignore */ }
  document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en'
}

export function t(zh, vars) {
  let text = lang.value === 'en' ? (EN[zh] ?? zh) : zh
  if (vars) text = text.replace(/\{(\w+)\}/g, (_, key) => (vars[key] ?? ''))
  return text
}

document.documentElement.lang = lang.value === 'zh' ? 'zh-CN' : 'en'
