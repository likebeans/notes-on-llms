<script setup lang="ts">
import { computed } from 'vue'
import { MODULE_DEFINITIONS } from '../../../config/modules'
import type { ContentIndexItem } from '../../../content/model'
import ArticleMeta from './ArticleMeta.vue'
import { data as content } from '../../../content/content.data'
import { resolvePrerequisites } from '../../../content/discovery'

const props = defineProps<{ page: ContentIndexItem }>()

const moduleLabel = computed(() => (
  props.page.module === 'site' ? '知识库' : MODULE_DEFINITIONS[props.page.module].title
))

const statusLabel = computed(() => ({
  verified: '已核验',
  'needs-review': '待复核',
  opinion: '观点',
  historical: '历史资料',
  draft: '草稿',
})[props.page.contentStatus])
</script>

<template>
  <header class="nl-article-intro">
    <div class="nl-article-kicker">
      <span>{{ moduleLabel }}</span>
      <span class="nl-article-status" :data-status="page.contentStatus">{{ statusLabel }}</span>
    </div>
    <h1>{{ page.title }}</h1>
    <p class="nl-article-description">{{ page.description }}</p>
    <ArticleMeta
      :level="page.level"
      :reading-time="page.readingTime"
      :prerequisites="resolvePrerequisites(page.prerequisites ?? [], content)"
      :tech-version="page.techVersion"
    />
  </header>
</template>
