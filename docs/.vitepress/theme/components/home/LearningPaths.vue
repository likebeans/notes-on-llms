<script setup lang="ts">
import { withBase } from 'vitepress'

const paths = [
  {
    title: '应用开发者',
    duration: '3–6 个月',
    description: '面向构建可靠 LLM 应用的工程实践，从提示、检索到工具调用与部署。',
    stages: ['Prompt 与 API', 'RAG 系统', 'Agent 与 MCP'],
    href: '/guide/roadmap#路径一-应用开发者-3-6-个月',
  },
  {
    title: '算法工程师',
    duration: '6–12 个月',
    description: '从应用基础延展到数据、微调、对齐和推理优化，建立模型定制能力。',
    stages: ['应用基础', '训练与对齐', '评估与推理'],
    href: '/guide/roadmap#路径二-算法工程师-6-12-个月',
  },
  {
    title: '面试冲刺',
    duration: '4–8 周',
    description: '以知识体系为索引，集中复习模型、系统设计与工程判断题。',
    stages: ['基础概念', '系统设计', '题目演练'],
    href: '/guide/prerequisites',
  },
]
</script>

<template>
  <section id="learning-paths" class="nl-home-section nl-learning-paths" aria-labelledby="learning-paths-title">
    <header class="nl-home-section-heading">
      <p class="nl-home-eyebrow">学习路径</p>
      <div>
        <h2 id="learning-paths-title">按你的问题进入，而不是按目录迷路</h2>
        <p>每条路径都从已有的指南出发，再把知识模块作为下一步的可回访参考。</p>
      </div>
    </header>

    <div class="nl-path-grid">
      <a v-for="path in paths" :key="path.title" class="nl-path-card" :href="withBase(path.href)">
        <span class="nl-path-card-duration">{{ path.duration }}</span>
        <h3>{{ path.title }}</h3>
        <p>{{ path.description }}</p>
        <ol>
          <li v-for="stage in path.stages" :key="stage">{{ stage }}</li>
        </ol>
        <span class="nl-path-card-link">进入路径 <span aria-hidden="true">→</span></span>
      </a>
    </div>
  </section>
</template>
