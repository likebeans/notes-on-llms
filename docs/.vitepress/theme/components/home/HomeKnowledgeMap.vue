<script setup lang="ts">
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { MODULE_DEFINITIONS } from '../../../config/modules'

const modules = computed(() => Object.entries(MODULE_DEFINITIONS).map(([key, definition], index) => ({
  key,
  index: String(index + 1).padStart(2, '0'),
  ...definition,
})))
</script>

<template>
  <section id="knowledge-map" class="nl-home-section nl-knowledge-map" aria-labelledby="knowledge-map-title">
    <header class="nl-home-section-heading">
      <p class="nl-home-eyebrow">知识地图</p>
      <div>
        <h2 id="knowledge-map-title">六个模块，一条可组合的学习主线</h2>
        <p>从提示与检索开始，逐步进入智能体、协议、模型定制与多模态系统。</p>
      </div>
    </header>

    <div class="nl-knowledge-map-grid">
      <a
        v-for="module in modules"
        :key="module.key"
        class="nl-module-card"
        :href="withBase(module.path)"
      >
        <span class="nl-module-card-index">{{ module.index }}</span>
        <span class="nl-module-card-title">{{ module.title }}</span>
        <span class="nl-module-card-description">{{ module.shortDescription }}</span>
        <span class="nl-module-card-meta">{{ module.items.length }} 篇入口 <span aria-hidden="true">→</span></span>
      </a>
    </div>
  </section>
</template>
