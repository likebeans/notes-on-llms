<script setup lang="ts">
defineProps<{ items: string[] }>()
</script>

<template>
  <section v-if="items.length" class="nl-learning-objectives" aria-labelledby="learning-objectives-title">
    <p class="nl-section-label">学习导航</p>
    <h2 id="learning-objectives-title">完成本文后，你将能够</h2>
    <ol>
      <li v-for="item in items" :key="item">{{ item }}</li>
    </ol>
  </section>
</template>
