import { normalizeUrl } from '../../../content/selectors'

export interface LearningEntry { completed: boolean; bookmarked: boolean; visitedAt?: string }
export interface LearningRecords { version: 1; entries: Record<string, LearningEntry> }
export const STORAGE_KEY = 'notes-on-llms:learning:v1'
export const emptyRecords = (): LearningRecords => ({version:1, entries:{}})

function safePath(url: string): string | undefined {
  if (!url.startsWith('/') || url.startsWith('//') || /[\\?#\s]/.test(url)) return
  try {
    if (decodeURIComponent(url).split('/').some(part => part === '..' || part === '.')) return
  } catch { return }
  return normalizeUrl(url)
}
export function parseRecords(raw: string | null): LearningRecords {
  const result = emptyRecords()
  if (!raw || raw.length > 500_000) return result
  try {
    const parsed = JSON.parse(raw)
    if (parsed?.version !== 1 || !parsed.entries || typeof parsed.entries !== 'object' || Array.isArray(parsed.entries)) return result
    for (const [url, value] of Object.entries(parsed.entries).slice(0,2000)) {
      const path = safePath(url)
      if (!path || !value || typeof value !== 'object') continue
      const entry = value as Record<string,unknown>
      if (typeof entry.completed !== 'boolean' || typeof entry.bookmarked !== 'boolean') continue
      const visitedAt = typeof entry.visitedAt === 'string' && Number.isFinite(Date.parse(entry.visitedAt)) ? entry.visitedAt : undefined
      result.entries[path] = {completed:entry.completed, bookmarked:entry.bookmarked, ...(visitedAt ? {visitedAt} : {})}
    }
  } catch { /* Corrupt or unsupported storage starts an empty local notebook. */ }
  return result
}
export function updateRecord(records: LearningRecords, url: string, patch: Partial<LearningEntry>): LearningRecords {
  const path = safePath(url)
  if (!path) return records
  const existing = records.entries[path] ?? {completed:false,bookmarked:false}
  return {version:1, entries:{...records.entries, [path]:{...existing,...patch}}}
}

export interface RecordStorage { getItem(key: string): string | null; setItem(key: string, value: string): void }
export type PendingRecordPatches = Record<string, Partial<LearningEntry>>

// Replay only local fields that have not reached storage. Other tabs retain
// ownership of every field this tab has not changed.
export function mergePendingRecords(records: LearningRecords, pending: PendingRecordPatches): LearningRecords {
  return Object.entries(pending).reduce((result, [url, patch]) => updateRecord(result,url,patch),records)
}

export function persistRecord(
  storage: RecordStorage | undefined,
  current: LearningRecords,
  url: string,
  patch: Partial<LearningEntry>,
  pending: PendingRecordPatches = {},
) {
  const path = safePath(url)
  const queued = path ? {...pending,[path]:{...pending[path],...patch}} : pending
  let base = current
  try {
    if (!storage) throw new Error('Storage unavailable')
    // A component can be absent while another tab writes. Re-read on every save.
    base = parseRecords(storage.getItem(STORAGE_KEY))
    const records = mergePendingRecords(base,queued)
    storage.setItem(STORAGE_KEY,JSON.stringify(records))
    return {records,pending:{} as PendingRecordPatches,persistent:true}
  } catch {
    return {records:mergePendingRecords(base,queued),pending:queued,persistent:false}
  }
}
