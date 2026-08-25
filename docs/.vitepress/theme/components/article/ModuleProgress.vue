<script setup lang="ts">
import { computed } from 'vue'
import { MODULE_DEFINITIONS } from '../../../config/modules'
import type { ModuleKey } from '../../../content/model'
import { findModuleStep } from './moduleProgress'

const props = defineProps<{
  module: ModuleKey
  currentUrl: string
}>()

const definition = computed(() => (
  props.module === 'site' ? undefined : MODULE_DEFINITIONS[props.module]
))
const currentStep = computed(() => (
  definition.value ? findModuleStep(definition.value.items, props.currentUrl) : undefined
))
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
      :aria-valuemin="1"
      :aria-valuemax="definition.items.length"
      :aria-valuenow="currentStep"
      :aria-valuetext="`第 ${currentStep} 节，共 ${definition.items.length} 节`"
    >
      <span :style="{ width: `${(currentStep / definition.items.length) * 100}%` }" />
    </div>
    <small>第 {{ currentStep }} / {{ definition.items.length }} 节</small>
  </section>
</template>
