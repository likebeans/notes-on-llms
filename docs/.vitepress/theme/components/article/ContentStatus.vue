<script setup lang="ts">
import { computed } from 'vue'
import type { ContentStatus as ContentStatusType } from '../../../content/model'
import { getContentStatusCopy } from './contentStatusCopy'

const props = defineProps<{
  status: ContentStatusType
  reviewed?: string
  sourceCount: number
  reviewScope?: string
  exampleStatus?: 'not-run' | 'partial' | 'executed'
}>()

const executionLabels = { 'not-run': '本轮未实跑', partial: '部分示例已运行', executed: '配套示例已运行' } as const
const copy = computed(() => getContentStatusCopy(props.status))
</script>

<template>
  <aside class="nl-content-status" :data-status="props.status" aria-label="内容状态">
    <p>
      <strong>内容状态：{{ copy.label }}</strong>
      <span v-if="props.reviewed">资料复核：{{ props.reviewed }}</span>
      <span>资料链接：{{ props.sourceCount }} 项（去重）</span>
    </p>
    <p v-if="props.reviewScope">复核范围：{{ props.reviewScope }}</p>
    <p>代码验证：{{ props.exampleStatus ? executionLabels[props.exampleStatus] : '未单独登记，请以文内运行记录为准' }}</p>
    <p>{{ copy.guidance }}</p>
  </aside>
</template>
