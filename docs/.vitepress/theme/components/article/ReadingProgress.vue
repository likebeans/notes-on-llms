<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { onContentUpdated } from 'vitepress'

// Mermaid watches <html> attributes, so keep progress styles on this component.
const progress = ref(0)
let active = false
let frame: number | undefined
let observer: ResizeObserver | undefined
let observedContent: Element | null = null

const updateProgress = () => {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight
  progress.value = scrollable > 0 ? Math.min(100, Math.max(0, (window.scrollY / scrollable) * 100)) : 0
}

const scheduleUpdate = () => {
  if (!active || frame !== undefined) return
  frame = window.requestAnimationFrame(() => {
    frame = undefined
    if (active) updateProgress()
  })
}

const observeContent = () => {
  if (!observer) return
  // The progress widget lives in the aside; observe the separate content column
  // so percentage updates cannot trigger their own resize feedback loop.
  const content = document.querySelector('.VPDoc .content-container')
  if (content === observedContent) return
  observer.disconnect()
  observedContent = content
  if (content) observer.observe(content)
}

onContentUpdated(() => {
  if (!active) return
  observeContent()
  scheduleUpdate()
})

onMounted(() => {
  active = true
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(scheduleUpdate)
    observeContent()
  }
  scheduleUpdate()
  window.addEventListener('scroll', scheduleUpdate, { passive: true })
  window.addEventListener('resize', scheduleUpdate)
})

onUnmounted(() => {
  active = false
  window.removeEventListener('scroll', scheduleUpdate)
  window.removeEventListener('resize', scheduleUpdate)
  observer?.disconnect()
  if (frame !== undefined) window.cancelAnimationFrame(frame)
})
</script>

<template>
  <div
    class="nl-reading-progress"
    :style="{ '--nl-reading-progress': `${progress}%` }"
    role="progressbar"
    aria-label="阅读进度"
    aria-valuemin="0"
    aria-valuemax="100"
    :aria-valuenow="Math.round(progress)"
  >
    <span>阅读进度</span>
    <strong>{{ Math.round(progress) }}%</strong>
  </div>
</template>
