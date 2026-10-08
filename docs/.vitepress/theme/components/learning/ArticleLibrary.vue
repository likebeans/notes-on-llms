<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as content } from '../../../content/content.data'
import { discoverArticles, isMirror, LEVEL_LABELS, MODULE_LABELS } from '../../../content/discovery'
import { normalizeUrl } from '../../../content/selectors'
import { useLearningRecords } from './useLearningRecords'
import { useLibraryFilters } from './useLibraryFilters'

const { query, module, level, kind, saved, limit, reset } = useLibraryFilters()
const { records, ready, persistent } = useLearningRecords()
const articles = content.filter(page => page.pageType === 'article')
const recordFor = (url: string) => records.value.entries[normalizeUrl(url)]
const completed = computed(() => articles.filter(page => recordFor(page.url)?.completed))
const bookmarked = computed(() => articles.filter(page => recordFor(page.url)?.bookmarked))
const recent = computed(() => articles.filter(page => recordFor(page.url)?.visitedAt)
  .sort((a,b) => (recordFor(b.url)?.visitedAt ?? '').localeCompare(recordFor(a.url)?.visitedAt ?? ''))[0])
const results = computed(() => discoverArticles(content,{
  query:query.value,module:module.value,level:level.value,kind:kind.value,
  savedUrls:saved.value === 'bookmarked' ? bookmarked.value.map(p=>p.url)
    : saved.value === 'completed' ? completed.value.map(p=>p.url) : undefined,
}))
const visible = computed(() => results.value.slice(0,limit.value))
</script>

<template>
  <section class="nl-library-notebook" aria-labelledby="notebook-title">
    <div>
      <p class="nl-section-label">个人学习记录</p>
      <h2 id="notebook-title">每次读一点，留下一条线索。</h2>
      <p v-if="ready && recent">上次阅读 <a :href="withBase(recent.url)">{{ recent.title }} →</a></p>
      <p v-else>读文章时可以标记完成或收藏，下一次从这里继续。</p>
      <small>{{ persistent ? '保存在当前浏览器，不跨设备同步；清除浏览器数据会移除记录。' : '浏览器存储不可用，记录仅在本次浏览期间保留。' }}</small>
    </div>
    <div class="nl-library-stats" aria-label="学习记录汇总">
      <button type="button" :aria-pressed="saved==='completed'" @click="saved=saved==='completed'?'all':'completed'"><strong>{{ ready ? completed.length : '—' }}</strong><span>已读文章</span></button>
      <button type="button" :aria-pressed="saved==='bookmarked'" @click="saved=saved==='bookmarked'?'all':'bookmarked'"><strong>{{ ready ? bookmarked.length : '—' }}</strong><span>我的收藏</span></button>
    </div>
  </section>

  <section class="nl-library" aria-labelledby="article-library-title">
    <h2 id="article-library-title">找到下一篇值得读的文章</h2>
    <p class="nl-library-hint">这里匹配标题、简介和标签。查找正文中的具体句子，请使用顶部的“搜索文档”。</p>
    <form class="nl-library-filters" role="search" aria-label="文章筛选" @submit.prevent>
      <label class="nl-library-query">主题关键词
        <input v-model="query" type="search" placeholder="例如：检索、Agent、LoRA" autocomplete="off" maxlength="200">
      </label>
      <label>模块<select v-model="module"><option value="all">全部模块</option><option v-for="(label,key) in MODULE_LABELS" :key="key" :value="key">{{ label }}</option></select></label>
      <label>难度<select v-model="level"><option value="all">全部难度</option><option v-for="(label,key) in LEVEL_LABELS" :key="key" :value="key">{{ label }}</option></select></label>
      <label>文章类型<select v-model="kind"><option value="all">全部类型</option><option value="mainline">主线文章</option><option value="mirror">镜像原文</option></select></label>
      <label>学习记录<select v-model="saved"><option value="all">全部文章</option><option value="bookmarked">我的收藏</option><option value="completed">已读文章</option></select></label>
    </form>
    <div class="nl-library-result-bar"><p role="status" aria-live="polite">找到 {{ results.length }} 篇文章</p><button type="button" @click="reset">重置筛选</button></div>
    <ul v-if="visible.length" class="nl-library-results">
      <li v-for="article in visible" :key="article.url">
        <div class="nl-library-labels"><span>{{ MODULE_LABELS[article.module] }}</span><span v-if="article.level">{{ LEVEL_LABELS[article.level] }}</span><span>{{ isMirror(article) ? '镜像原文' : '主线文章' }}</span><span v-if="recordFor(article.url)?.completed">✓ 已读</span><span v-if="recordFor(article.url)?.bookmarked">★ 已收藏</span></div>
        <h3><a :href="withBase(article.url)">{{ article.title }} <span aria-hidden="true">↗</span></a></h3>
        <p>{{ article.description }}</p>
        <small>约 {{ article.readingTime }} 分钟 · 更新于 {{ article.updated }}</small>
      </li>
    </ul>
    <div v-else class="nl-library-empty"><strong>当前组合下没有文章</strong><p>试试减少关键词或取消一个筛选条件。收藏与已读记录需要先在文章页标记。</p></div>
    <button v-if="visible.length < results.length" type="button" class="nl-library-more" @click="limit += 20">再显示 20 篇（还有 {{ results.length - visible.length }} 篇）</button>
  </section>
</template>
