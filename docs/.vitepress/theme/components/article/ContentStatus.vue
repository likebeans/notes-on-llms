<script setup lang="ts">
import type { ContentStatus as ContentStatusType } from '../../../content/model'

const props = defineProps<{
  status: ContentStatusType
  reviewed?: string
  sourceCount: number
}>()

const labels: Record<ContentStatusType, string> = {
  verified: '已核验',
  'needs-review': '待复核',
  opinion: '观点内容',
  historical: '历史资料',
  draft: '草稿',
}
</script>

<template>
  <aside class="nl-content-status" :data-status="props.status" aria-label="内容状态">
    <p>
      <strong>内容状态：{{ labels[props.status] }}</strong>
      <span v-if="props.reviewed">最近复核：{{ props.reviewed }}</span>
      <span>参考资料：{{ props.sourceCount }} 项</span>
    </p>
  </aside>
</template>
