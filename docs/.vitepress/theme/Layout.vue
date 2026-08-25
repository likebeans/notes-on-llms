<script setup lang="ts">
import DefaultTheme from 'vitepress/theme-without-fonts'
import { computed } from 'vue'
import { useData, useRoute, withBase } from 'vitepress'
import { data as content } from '../content/content.data'
import { findAdjacentArticle, normalizeUrl } from '../content/selectors'
import ArticleIntro from './components/article/ArticleIntro.vue'
import ContentStatus from './components/article/ContentStatus.vue'
import DraftNotice from './components/article/DraftNotice.vue'
import ModuleProgress from './components/article/ModuleProgress.vue'
import NextStep from './components/article/NextStep.vue'
import ReadingProgress from './components/article/ReadingProgress.vue'

const { Layout } = DefaultTheme
const { frontmatter } = useData()
const route = useRoute()
const routeUrl = computed(() => normalizeUrl(route.path))
const page = computed(() => content.find(item => (
  normalizeUrl(item.url) === routeUrl.value
  || normalizeUrl(withBase(item.url)) === routeUrl.value
)))
const currentUrl = computed(() => page.value?.url ?? route.path)
const adjacent = computed(() => findAdjacentArticle(content, currentUrl.value))
</script>

<template>
  <div :class="{ 'nl-article-page': frontmatter.pageType === 'article' }">
    <Layout>
      <template #doc-before>
        <DraftNotice v-if="frontmatter.contentStatus === 'draft'" />
        <ArticleIntro v-if="page && frontmatter.pageType === 'article'" :page="page" />
      </template>
      <template #sidebar-nav-before>
        <ModuleProgress
          v-if="page?.module && page.module !== 'site'"
          :module="page.module"
          :current-url="currentUrl"
        />
      </template>
      <template #aside-outline-before>
        <ReadingProgress v-if="frontmatter.pageType === 'article'" />
      </template>
      <template #doc-after>
        <ContentStatus
          v-if="page && frontmatter.pageType === 'article'"
          :status="page.contentStatus"
          :reviewed="page.reviewed"
          :source-count="page.sourceCount"
        />
        <NextStep v-if="frontmatter.pageType === 'article'" v-bind="adjacent" />
      </template>
    </Layout>
  </div>
</template>
