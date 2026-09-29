<template>
  <button type="button" class="lang-switch" :title="lang === 'zh' ? 'Switch to English' : '切换到中文'" @click="setLang(lang === 'zh' ? 'en' : 'zh')">
    <span :class="{ on: lang === 'zh' }">中</span><span :class="{ on: lang === 'en' }">EN</span>
  </button>
</template>

<script setup>
import { lang, setLang } from '../lib/i18n.js'
</script>

<style>
.lang-switch { display: inline-flex; align-items: center; gap: 2px; padding: 2px; border-radius: 999px; border: 1px solid currentColor; background: transparent; color: inherit; font: inherit; font-size: 12px; line-height: 1; cursor: pointer; opacity: .85; }
.lang-switch span { padding: 4px 8px; border-radius: 999px; }
.lang-switch span.on { background: rgba(127, 127, 127, .3); font-weight: 700; }
</style>
