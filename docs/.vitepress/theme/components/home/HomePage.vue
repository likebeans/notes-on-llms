<script setup lang="ts">
import { withBase } from 'vitepress'
import HomeKnowledgeMap from './HomeKnowledgeMap.vue'
import LearningPaths from './LearningPaths.vue'
import RecentUpdates from './RecentUpdates.vue'
import SiteIntroduction from './SiteIntroduction.vue'
</script>

<template>
  <main class="nl-home-page">
    <section class="nl-home-hero" aria-labelledby="home-title">
      <div class="nl-home-hero-copy">
        <p class="nl-home-eyebrow">研究手册 · 知识地图</p>
        <h1 id="home-title">从模型原理到可运行的智能系统</h1>
        <p class="nl-home-lede">
          Notes on LLMs 是一份持续核验的大模型研究手册：把每篇笔记放进“场景、机制、评估、风险”的框架里，连接概念、工程选择与实践问题。
        </p>
        <div class="nl-home-actions" aria-label="开始学习">
          <a class="nl-home-action nl-home-action-primary" :href="withBase('/guide/roadmap')">查看学习路线</a>
          <a class="nl-home-action" :href="withBase('/llms/')">浏览知识体系</a>
        </div>
      </div>
      <nav class="nl-home-hero-index" aria-label="首页章节">
        <p>本页索引</p>
        <ol>
          <li><a :href="withBase('/#knowledge-map')">六个学习模块</a></li>
          <li><a :href="withBase('/#learning-paths')">三条学习路径</a></li>
          <li><a :href="withBase('/#recent-updates')">最近更新</a></li>
        </ol>
      </nav>
    </section>

    <HomeKnowledgeMap />
    <LearningPaths />
    <RecentUpdates />
    <SiteIntroduction />
  </main>
</template>
