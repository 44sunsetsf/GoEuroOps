<template>
  <component :is="view" />
</template>

<script setup>
import { defineAsyncComponent } from 'vue'

// Students land on the public site; the studio console lives at /studio.
// Each side is its own chunk, so their global styles never meet.
const isStudio = window.location.pathname.replace(/\/+$/, '').endsWith('/studio')
const view = defineAsyncComponent(isStudio ? () => import('./App.vue') : () => import('./student/StudentSite.vue'))
</script>
