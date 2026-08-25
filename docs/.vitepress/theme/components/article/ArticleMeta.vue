<script setup lang="ts">
import { withBase } from 'vitepress'

const props = defineProps<{
  level?: 'beginner' | 'intermediate' | 'advanced'
  readingTime: number
  prerequisites?: string[]
  techVersion?: string
}>()

const levelLabel = {
  beginner: '入门',
  intermediate: '进阶',
  advanced: '高级',
} as const

const isExternal = (href: string) => /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href)
const toHref = (href: string) => isExternal(href) ? href : withBase(href)
</script>

<template>
  <dl class="nl-article-meta" aria-label="文章信息">
    <div v-if="props.level">
      <dt>难度</dt>
      <dd>{{ levelLabel[props.level] }}</dd>
    </div>
    <div>
      <dt>阅读</dt>
      <dd>{{ props.readingTime }} 分钟</dd>
    </div>
    <div v-if="props.techVersion">
      <dt>版本</dt>
      <dd>{{ props.techVersion }}</dd>
    </div>
    <div v-if="props.prerequisites?.length">
      <dt>前置</dt>
      <dd>
        <a
          v-for="prerequisite in props.prerequisites"
          :key="prerequisite"
          :href="toHref(prerequisite)"
          :target="isExternal(prerequisite) ? '_blank' : undefined"
          :rel="isExternal(prerequisite) ? 'noreferrer' : undefined"
        >
          {{ prerequisite }}
        </a>
      </dd>
    </div>
  </dl>
</template>
