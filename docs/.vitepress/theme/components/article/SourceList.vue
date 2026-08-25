<script setup lang="ts">
import { withBase } from 'vitepress'

interface SourceItem {
  title: string
  href: string
  note?: string
}

defineProps<{ items: SourceItem[] }>()

const isExternal = (href: string) => /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href)
const toHref = (href: string) => isExternal(href) ? href : withBase(href)
</script>

<template>
  <section v-if="items.length" class="nl-source-list" aria-labelledby="source-list-title">
    <p class="nl-section-label">延伸阅读</p>
    <h2 id="source-list-title">参考资料</h2>
    <ul>
      <li v-for="item in items" :key="item.href">
        <a
          :href="toHref(item.href)"
          :target="isExternal(item.href) ? '_blank' : undefined"
          :rel="isExternal(item.href) ? 'noreferrer' : undefined"
        >{{ item.title }}</a>
        <span v-if="item.note"> — {{ item.note }}</span>
      </li>
    </ul>
  </section>
</template>
