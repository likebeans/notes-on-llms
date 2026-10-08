<script setup lang="ts">
import { computed } from 'vue'
import { MODULE_DEFINITIONS } from '../../../config/modules'
import type { ModuleKey } from '../../../content/model'
import { findModuleStep } from './moduleProgress'
import { normalizeUrl } from '../../../content/selectors'
import { useLearningRecords } from '../learning/useLearningRecords'

const props = defineProps<{
  module: ModuleKey
  currentUrl: string
}>()

const { records, ready } = useLearningRecords()
const definition = computed(() => (
  props.module === 'site' ? undefined : MODULE_DEFINITIONS[props.module]
))
const currentStep = computed(() => (
  definition.value ? findModuleStep(definition.value.items, props.currentUrl) : undefined
))
const learnable = computed(() => definition.value?.items.filter(item => normalizeUrl(item.link) !== normalizeUrl(definition.value!.path)) ?? [])
const completedCount = computed(() => learnable.value.filter(item => records.value.entries[normalizeUrl(item.link)]?.completed).length)
</script>

<template>
  <section
    v-if="definition && currentStep !== undefined"
    class="nl-module-progress"
    :aria-label="`${definition.title} 学习进度`"
  >
    <p>{{ definition.title }} 学习路径</p>
    <div
      class="nl-module-progress-track"
      role="progressbar"
      :aria-valuemin="0"
      :aria-valuemax="learnable.length"
      :aria-valuenow="completedCount"
      :aria-valuetext="`已完成 ${completedCount} 篇，共 ${learnable.length} 篇`"
    >
      <span :style="{ width: `${learnable.length ? (completedCount / learnable.length) * 100 : 0}%` }" />
    </div>
    <small>当前位置：第 {{ currentStep }} / {{ definition.items.length }} 节</small>
    <small v-if="ready">已完成 {{ completedCount }} / {{ learnable.length }} 篇</small>
  </section>
</template>
