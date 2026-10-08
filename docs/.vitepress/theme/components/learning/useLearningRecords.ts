import { onMounted, onUnmounted, readonly, ref } from 'vue'
import { emptyRecords, parseRecords, STORAGE_KEY, persistRecord, mergePendingRecords, type PendingRecordPatches, type LearningEntry } from './records'

const records = ref(emptyRecords())
const ready = ref(false)
const persistent = ref(true)
const pending = ref<PendingRecordPatches>({})

export function useLearningRecords() {
  const refresh = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY || event.key === null) {
      records.value = mergePendingRecords(parseRecords(event.newValue),pending.value)
    }
  }
  onMounted(() => {
    try {
      records.value = mergePendingRecords(parseRecords(localStorage.getItem(STORAGE_KEY)),pending.value)
      persistent.value = Object.keys(pending.value).length === 0
    }
    catch { persistent.value = false }
    ready.value = true
    window.addEventListener('storage',refresh)
  })
  onUnmounted(() => {
    if (typeof window !== 'undefined') window.removeEventListener('storage',refresh)
  })
  function save(url: string, patch: Partial<LearningEntry>) {
    if (!ready.value) return
    let storage: Storage | undefined
    try { storage = window.localStorage } catch { /* Security settings may block access itself. */ }
    const result = persistRecord(storage,records.value,url,patch,pending.value)
    records.value = result.records
    pending.value = result.pending
    persistent.value = result.persistent
  }
  return {records:readonly(records),ready:readonly(ready),persistent:readonly(persistent),save}
}
