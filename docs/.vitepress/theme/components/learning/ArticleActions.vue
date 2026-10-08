<script setup lang="ts">
import { computed, watch } from 'vue'
import { withBase } from 'vitepress'
import { normalizeUrl } from '../../../content/selectors'
import { useLearningRecords } from './useLearningRecords'

const props = defineProps<{ url: string }>()
const { records, ready, persistent, save } = useLearningRecords()
const entry = computed(() => records.value.entries[normalizeUrl(props.url)])
watch([() => props.url, ready], ([url, isReady]) => {
  if (isReady) save(url,{visitedAt:new Date().toISOString()})
},{immediate:true})
</script>

<template>
  <div class="nl-study-actions" aria-label="学习记录">
    <div class="nl-study-buttons">
      <button type="button" :disabled="!ready" :aria-pressed="!!entry?.completed" @click="save(url,{completed:!entry?.completed})">
        {{ entry?.completed ? '✓ 已完成阅读' : '标记为已读' }}
      </button>
      <button type="button" :disabled="!ready" :aria-pressed="!!entry?.bookmarked" @click="save(url,{bookmarked:!entry?.bookmarked})">
        {{ entry?.bookmarked ? '★ 已收藏' : '收藏这篇' }}
      </button>
      <a :href="withBase('/guide/library')">我的学习与文章检索 →</a>
    </div>
    <small role="status">{{ persistent ? '学习记录仅保存在当前浏览器，可随时取消标记。' : '浏览器存储不可用，当前记录仅在本次浏览期间保留。' }}</small>
  </div>
</template>
