<script setup lang="ts">
import { useData } from 'vitepress'

const { frontmatter } = useData()
</script>

<template>
  <aside v-if="frontmatter.contentStatus === 'draft'" class="nl-draft-notice" role="status">
    <strong>草稿：不会进入生产发布</strong>
  </aside>
</template>
