import { describe, expect, it } from 'vitest'
import { emptyRecords, parseRecords, updateRecord } from '../../docs/.vitepress/theme/components/learning/records'

describe('browser learning records', () => {
  it('recovers from corrupt or obsolete storage', () => {
    for (const raw of [null, '{bad', 'null', '{"version":99,"entries":{}}']) {
      expect(parseRecords(raw)).toEqual(emptyRecords())
    }
  })
  it('restores valid entries while rejecting external paths and invalid fields', () => {
    const result = parseRecords(JSON.stringify({version:1, entries:{
      '/llms/rag/index.html':{completed:true, bookmarked:false, visitedAt:'2026-10-08T00:00:00.000Z'},
      '//evil.example/path':{completed:true},
      '/bad':{completed:'yes',bookmarked:true},
      '/guide/../secrets':{completed:true,bookmarked:true},
    }}))
    expect(Object.keys(result.entries)).toEqual(['/llms/rag'])
    expect(result.entries['/llms/rag'].completed).toBe(true)
  })
  it('changes one action without losing saved state and can undo completion', () => {
    const initial = updateRecord(emptyRecords(), '/llms/rag/chunking', {bookmarked:true})
    const done = updateRecord(initial, '/llms/rag/chunking.html', {completed:true})
    const undone = updateRecord(done, '/llms/rag/chunking', {completed:false})
    expect(initial.entries['/llms/rag/chunking'].completed).toBe(false)
    expect(done.entries['/llms/rag/chunking']).toMatchObject({completed:true,bookmarked:true})
    expect(undone.entries['/llms/rag/chunking']).toMatchObject({completed:false,bookmarked:true})
    expect(updateRecord(initial, 'javascript:alert(1)', {completed:true})).toEqual(initial)
  })
})

import { persistRecord, mergePendingRecords, STORAGE_KEY } from '../../docs/.vitepress/theme/components/learning/records'

describe('saving across browser tabs', () => {
  it('merges a visit into the latest stored records instead of overwriting another tab', () => {
    const stale = updateRecord(emptyRecords(), '/llms/rag/chunking', {completed:true})
    const latest = updateRecord(stale, '/llms/agent/memory', {bookmarked:true})
    const store = new Map([[STORAGE_KEY,JSON.stringify(latest)]])
    const storage = {getItem:(key:string)=>store.get(key)??null,setItem:(key:string,value:string)=>{store.set(key,value)}}
    const result = persistRecord(storage,stale,'/llms/rag/chunking',{visitedAt:'2026-10-08T00:00:00.000Z'})
    expect(result.records.entries['/llms/agent/memory'].bookmarked).toBe(true)
    expect(parseRecords(store.get(STORAGE_KEY)!).entries['/llms/rag/chunking'].completed).toBe(true)
    expect(result.persistent).toBe(true)
  })
  it('keeps the current session usable when storage is denied or full', () => {
    const denied = {getItem:()=>{throw new Error('denied')},setItem:()=>{throw new Error('denied')}}
    const result = persistRecord(denied,emptyRecords(),'/llms/rag/chunking',{bookmarked:true})
    expect(result.persistent).toBe(false)
    expect(result.records.entries['/llms/rag/chunking'].bookmarked).toBe(true)
  })
})

describe('unsaved changes after storage quota failures', () => {
  function quotaStorage(initial = emptyRecords()) {
    let raw = JSON.stringify(initial)
    let full = true
    return {
      storage: {
        getItem: () => raw,
        setItem: (_key: string, value: string) => {
          if (full) throw new Error('QuotaExceededError')
          raw = value
        },
      },
      recover: () => { full = false },
      externalWrite: (records: ReturnType<typeof emptyRecords>) => { raw = JSON.stringify(records) },
      read: () => parseRecords(raw),
    }
  }

  it('retains two failed edits to different articles and persists both after recovery', () => {
    const store = quotaStorage()
    const first = persistRecord(store.storage,emptyRecords(),'/llms/agent/memory',{bookmarked:true})
    const second = persistRecord(store.storage,first.records,'/llms/rag/chunking',{completed:true},first.pending)
    expect(second.records.entries['/llms/agent/memory']?.bookmarked).toBe(true)
    expect(second.records.entries['/llms/rag/chunking']?.completed).toBe(true)
    expect(second.persistent).toBe(false)
    store.recover()
    const recovered = persistRecord(store.storage,second.records,'/llms/rag/chunking',{visitedAt:'2026-10-08T00:00:00.000Z'},second.pending)
    expect(recovered.persistent).toBe(true)
    expect(recovered.pending).toEqual({})
    expect(store.read().entries['/llms/agent/memory']?.bookmarked).toBe(true)
    expect(store.read().entries['/llms/rag/chunking']?.completed).toBe(true)
  })

  it('retains different unsaved fields on the same article', () => {
    const store = quotaStorage()
    const first = persistRecord(store.storage,emptyRecords(),'/llms/agent/memory',{bookmarked:true})
    const second = persistRecord(store.storage,first.records,'/llms/agent/memory',{completed:true},first.pending)
    expect(second.records.entries['/llms/agent/memory']).toMatchObject({bookmarked:true,completed:true})
    store.recover()
    const recovered = persistRecord(store.storage,second.records,'/llms/agent/memory',{visitedAt:'2026-10-08T00:00:00.000Z'},second.pending)
    expect(store.read().entries['/llms/agent/memory']).toMatchObject({bookmarked:true,completed:true})
    expect(recovered.pending).toEqual({})
  })

  it('preserves pending fields when remounts or storage events refresh the saved snapshot', () => {
    const store = quotaStorage()
    const failed = persistRecord(store.storage,emptyRecords(),'/llms/agent/memory',{bookmarked:true})
    const refreshed = updateRecord(emptyRecords(),'/llms/agent/memory',{completed:true})
    const merged = mergePendingRecords(refreshed,failed.pending)
    expect(merged.entries['/llms/agent/memory']).toMatchObject({bookmarked:true,completed:true})
    expect(refreshed.entries['/llms/agent/memory'].bookmarked).toBe(false)
    const clearedElsewhere = mergePendingRecords(emptyRecords(),failed.pending)
    expect(clearedElsewhere.entries['/llms/agent/memory']).toMatchObject({bookmarked:true,completed:false})
  })

  it('replays only unsaved fields onto external storage changes and releases them after saving', () => {
    const initial = updateRecord(emptyRecords(),'/llms/agent/memory',{bookmarked:true})
    const store = quotaStorage(initial)
    const failed = persistRecord(store.storage,initial,'/llms/agent/memory',{completed:true})
    const latest = updateRecord(updateRecord(initial,'/llms/agent/memory',{bookmarked:false}),'/llms/rag/chunking',{bookmarked:true})
    store.externalWrite(latest)
    store.recover()
    const recovered = persistRecord(store.storage,failed.records,'/llms/agent/memory',{visitedAt:'2026-10-08T00:00:00.000Z'},failed.pending)
    expect(recovered.records.entries['/llms/agent/memory']).toMatchObject({bookmarked:false,completed:true})
    expect(recovered.records.entries['/llms/rag/chunking'].bookmarked).toBe(true)
    store.externalWrite(updateRecord(store.read(),'/llms/agent/memory',{completed:false}))
    const next = persistRecord(store.storage,recovered.records,'/llms/rag/chunking',{completed:true},recovered.pending)
    expect(next.records.entries['/llms/agent/memory'].completed).toBe(false)
  })
})
