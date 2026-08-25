import { isDraft, type ContentIndexItem } from './model'

export interface AdjacentArticles {
  previous?: ContentIndexItem
  next?: ContentIndexItem
}

export const normalizeUrl = (url: string): string => {
  const path = url.replace(/^\/+/, '').replace(/\/+$/, '')
  return path ? `/${path}` : '/'
}

export function selectRecentUpdates(items: ContentIndexItem[], limit = 4): ContentIndexItem[] {
  return items
    .filter(item => !isDraft(item))
    .sort((a, b) => b.updated.localeCompare(a.updated) || a.title.localeCompare(b.title, 'zh-CN'))
    .slice(0, Math.max(0, limit))
}

const compareOrder = (a: ContentIndexItem, b: ContentIndexItem): number => {
  const aHasOrder = typeof a.order === 'number' && Number.isFinite(a.order)
  const bHasOrder = typeof b.order === 'number' && Number.isFinite(b.order)

  if (aHasOrder && bHasOrder && a.order !== b.order) return a.order! - b.order!
  if (aHasOrder !== bHasOrder) return aHasOrder ? -1 : 1

  return a.title.localeCompare(b.title, 'zh-CN') || normalizeUrl(a.url).localeCompare(normalizeUrl(b.url))
}

export function findAdjacentArticle(items: ContentIndexItem[], url: string): AdjacentArticles {
  const currentUrl = normalizeUrl(url)
  const current = items.find(item => normalizeUrl(item.url) === currentUrl && !isDraft(item))
  if (!current) return {}

  const moduleArticles = items
    .filter(item => item.module === current.module && item.pageType === 'article' && !isDraft(item))
    .sort(compareOrder)
  const index = moduleArticles.findIndex(item => normalizeUrl(item.url) === currentUrl)
  if (index < 0) return {}

  const adjacent: AdjacentArticles = {}
  if (index > 0) adjacent.previous = moduleArticles[index - 1]
  if (index < moduleArticles.length - 1) adjacent.next = moduleArticles[index + 1]
  return adjacent
}
