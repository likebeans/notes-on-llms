import type { ContentIndexItem, ModuleKey } from './model'
import { normalizeUrl } from './selectors'

export const MODULE_LABELS: Record<ModuleKey, string> = {
  rag:'RAG', agent:'Agent', mcp:'MCP', prompt:'Prompt', training:'训练微调', multimodal:'多模态', site:'学习与实践',
}
export const LEVEL_LABELS = { beginner:'入门', intermediate:'进阶', advanced:'高级' } as const
export const isMirror = (item: ContentIndexItem) => item.tags.includes('csdn-mirror') || item.url.includes('/csdn/')
export interface DiscoveryFilters {
  query: string
  module: string
  level: string
  kind: string
  savedUrls?: string[]
}
export function discoverArticles(items: ContentIndexItem[], filters: DiscoveryFilters): ContentIndexItem[] {
  const terms = filters.query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  const saved = filters.savedUrls ? new Set(filters.savedUrls.map(normalizeUrl)) : undefined
  return items.filter(item => {
    if (item.pageType !== 'article' || item.contentStatus === 'draft') return false
    if (filters.module !== 'all' && item.module !== filters.module) return false
    if (filters.level !== 'all' && item.level !== filters.level) return false
    if (filters.kind === 'mainline' && isMirror(item)) return false
    if (filters.kind === 'mirror' && !isMirror(item)) return false
    if (saved && !saved.has(normalizeUrl(item.url))) return false
    const haystack = [item.title,item.description,MODULE_LABELS[item.module],...item.tags].join(' ').toLocaleLowerCase()
    return terms.every(term => haystack.includes(term))
  }).sort((a,b) => b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title,'zh-CN'))
}
export function resolvePrerequisites(links: string[], items: ContentIndexItem[]) {
  return links.map(href => {
    const page = items.find(item => normalizeUrl(item.url) === normalizeUrl(href.split(/[?#]/)[0]))
    return { href, title: page?.title ?? (/^https?:/.test(href) ? '外部前置资料' : '延伸阅读') }
  })
}
