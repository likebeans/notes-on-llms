<script setup lang="ts">
import { withBase } from 'vitepress'
import type { ContentIndexItem } from '../../../content/model'

defineProps<{
  previous?: ContentIndexItem
  next?: ContentIndexItem
}>()

const toHref = (href: string) => withBase(href)
</script>

<template>
  <nav v-if="previous || next" class="nl-next-step" aria-label="继续学习">
    <p class="nl-section-label">继续学习</p>
    <div>
      <a v-if="previous" class="nl-next-step-previous" :href="toHref(previous.url)">
        <span>← 上一篇</span>
        <strong>{{ previous.title }}</strong>
      </a>
      <a v-if="next" class="nl-next-step-next" :href="toHref(next.url)">
        <span>下一篇 →</span>
        <strong>{{ next.title }}</strong>
      </a>
    </div>
  </nav>
</template>
