<script setup lang="ts">
import { computed } from 'vue'
import type { ContentStatus as ContentStatusType } from '../../../content/model'
import { getContentStatusCopy } from './contentStatusCopy'

const props = defineProps<{
  status: ContentStatusType
  reviewed?: string
  sourceCount: number
}>()

const copy = computed(() => getContentStatusCopy(props.status))
</script>

<template>
  <aside class="nl-content-status" :data-status="props.status" aria-label="内容状态">
    <p>
      <strong>内容状态：{{ copy.label }}</strong>
      <span v-if="props.reviewed">最近复核：{{ props.reviewed }}</span>
      <span>参考资料：{{ props.sourceCount }} 项</span>
    </p>
    <p>{{ copy.guidance }}</p>
  </aside>
</template>
