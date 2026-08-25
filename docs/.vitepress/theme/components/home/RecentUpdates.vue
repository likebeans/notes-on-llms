<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as content } from '../../../content/content.data'
import { selectRecentUpdates } from '../../../content/selectors'
import { MODULE_DEFINITIONS } from '../../../config/modules'
import type { ContentStatus, ModuleKey } from '../../../content/model'

const recentUpdates = computed(() => selectRecentUpdates(content, 4))

const moduleLabel = (module: ModuleKey): string => (
  module === 'site' ? '指南' : MODULE_DEFINITIONS[module].title
)

const statusLabel = (status: ContentStatus): string => ({
  verified: '已核验',
  'needs-review': '待复核',
  opinion: '作者观点',
  historical: '历史资料',
  draft: '草稿',
}[status])
</script>

<template>
  <section id="recent-updates" class="nl-home-section nl-recent-updates" aria-labelledby="recent-updates-title">
    <header class="nl-home-section-heading nl-home-section-heading-row">
      <div>
        <p class="nl-home-eyebrow">构建时内容索引</p>
        <h2 id="recent-updates-title">最近更新</h2>
      </div>
      <a :href="withBase('/about/changelog')">查看更新记录 <span aria-hidden="true">→</span></a>
    </header>

    <ol class="nl-update-list">
      <li v-for="page in recentUpdates" :key="page.url">
        <time :datetime="page.updated">{{ page.updated }}</time>
        <span class="nl-update-module">{{ moduleLabel(page.module) }}</span>
        <a :href="withBase(page.url)">{{ page.title }}</a>
        <span class="nl-update-status" :data-status="page.contentStatus">{{ statusLabel(page.contentStatus) }}</span>
      </li>
    </ol>
  </section>
</template>
