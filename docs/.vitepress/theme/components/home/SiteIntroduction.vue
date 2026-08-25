<script setup lang="ts">
import { withBase } from 'vitepress'
</script>

<template>
  <section class="nl-home-section nl-site-introduction" aria-labelledby="site-introduction-title">
    <div>
      <p class="nl-home-eyebrow">写作说明</p>
      <h2 id="site-introduction-title">把每一篇笔记放回它解决的问题里</h2>
    </div>
    <div class="nl-site-introduction-copy">
      <p>
        本站由 likebeans 维护，记录学习、研究与构建 LLM 系统时反复遇到的概念和取舍。内容不是追逐单点热点的清单，而是可回看的工程判断。
      </p>
      <p>
        已核验内容标注复核时间与来源数量；仍待复核、历史资料和作者观点会明确说明，方便你决定如何使用。
      </p>
      <p class="nl-site-introduction-links">
        <a :href="withBase('/about/')">了解本站</a>
        <a href="https://github.com/likebeans/notes-on-llms">在 GitHub 查看源文件 <span aria-hidden="true">↗</span></a>
      </p>
    </div>
  </section>
</template>
