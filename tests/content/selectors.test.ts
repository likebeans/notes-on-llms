import { describe, expect, it, vi } from 'vitest'
import type { ContentIndexItem } from '../../docs/.vitepress/content/model'
import { findAdjacentArticle, normalizeUrl, selectRecentUpdates } from '../../docs/.vitepress/content/selectors'

vi.mock('vitepress', () => ({
  createContentLoader: (_pattern: string, options: unknown) => options
}))

import loader from '../../docs/.vitepress/content/content.data'

const page = (overrides: Partial<ContentIndexItem>): ContentIndexItem => ({
  title: '文章',
  description: '测试文章',
  pageType: 'article',
  module: 'rag',
  level: 'intermediate',
  prerequisites: [],
  updated: '2026-08-20',
  reviewed: '2026-08-20',
  contentStatus: 'verified',
  techVersion: '2026',
  tags: ['test'],
  url: '/llms/rag/article',
  readingTime: 1,
  sourceCount: 0,
  ...overrides
})

const pages: ContentIndexItem[] = [
  page({ title: 'RAG', url: '/llms/rag/', order: 1, updated: '2026-08-23' }),
  page({ title: '检索', url: '/llms/rag/retrieval', order: 2, updated: '2026-08-25' }),
  page({ title: '重排序', url: '/llms/rag/rerank', order: 3, updated: '2026-08-24' }),
  page({ title: '草稿', url: '/llms/rag/draft', order: 4, updated: '2026-08-26', contentStatus: 'draft' }),
  page({ title: 'Agent', url: '/llms/agent/', module: 'agent', order: 1, updated: '2026-08-22' }),
]

describe('normalizeUrl', () => {
  it('normalizes leading and trailing slashes while preserving root', () => {
    expect(normalizeUrl('/llms/rag/')).toBe('/llms/rag')
    expect(normalizeUrl('llms/rag/')).toBe('/llms/rag')
    expect(normalizeUrl('/llms/rag/index.html')).toBe('/llms/rag')
    expect(normalizeUrl('/llms/rag/retrieval.html')).toBe('/llms/rag/retrieval')
    expect(normalizeUrl('/')).toBe('/')
  })
})

describe('selectRecentUpdates', () => {
  it('sorts by updated date, excludes drafts, and respects the limit', () => {
    expect(selectRecentUpdates(pages, 2).map(item => item.title)).toEqual(['检索', '重排序'])
  })
})

describe('findAdjacentArticle', () => {
  it('returns ordered neighbors from the current module and skips drafts', () => {
    const adjacent = findAdjacentArticle(pages, '/llms/rag/')
    expect(adjacent.previous).toBeUndefined()
    expect(adjacent.next?.url).toBe('/llms/rag/retrieval')
    expect(findAdjacentArticle(pages, '/llms/rag')).toEqual(adjacent)
    expect(findAdjacentArticle(pages, '/llms/rag/rerank').next).toBeUndefined()
  })

  it('does not cross module boundaries when finding neighbors', () => {
    expect(findAdjacentArticle(pages, '/llms/agent/').next).toBeUndefined()
    expect(findAdjacentArticle(pages, '/llms/rag/retrieval').previous?.url).toBe('/llms/rag/')
  })

  it('uses defined order values before deterministic tie breakers', () => {
    const unordered = [
      page({ title: '未排序 Z', url: '/llms/rag/z' }),
      page({ title: '有序', url: '/llms/rag/ordered', order: 1 }),
      page({ title: '未排序 A', url: '/llms/rag/a' }),
    ]
    expect(findAdjacentArticle(unordered, '/llms/rag/ordered').next?.title).toBe('未排序 A')
    expect(findAdjacentArticle(unordered, '/llms/rag/a').previous?.title).toBe('有序')
  })

  it('does not mutate the input collection', () => {
    const input = [...pages]
    const before = input.map(item => ({ ...item }))
    selectRecentUpdates(input, 2)
    findAdjacentArticle(input, '/llms/rag/')
    expect(input).toEqual(before)
  })
})

describe('content loader transform', () => {
  it('filters internal and draft pages, validates public metadata, and derives metrics', () => {
    type RawPage = { url: string; src: string; frontmatter: Record<string, unknown> }
    const article = (url: string, frontmatter: Record<string, unknown>, src: string): RawPage => ({ url, frontmatter, src })
    const publicFrontmatter = {
      title: '公开文章', description: '描述', pageType: 'article', module: 'rag', level: 'beginner',
      prerequisites: [], updated: '2026-08-25', reviewed: '2026-08-25', contentStatus: 'verified',
      techVersion: '2026', tags: ['test']
    }
    const transform = (loader as unknown as { transform(raw: RawPage[]): ContentIndexItem[] }).transform
    const longSource = `---\nignored: true\n---\n${'word '.repeat(700)}`
    const raw: RawPage[] = [
      article('/开发计划.html', {}, 'invalid internal content'),
      article('/superpowers/internal.html', {}, 'invalid internal content'),
      article('/llms/rag/draft.html', { ...publicFrontmatter, title: '草稿', contentStatus: 'draft' }, 'draft'),
      article('/llms/rag/long.html', publicFrontmatter, longSource),
      article('/llms/rag/references/index.html', { ...publicFrontmatter, title: '参考资料' }, '正文\n\n## 参考资料\n- 第一项\n* 第二项')
    ]

    const result = transform(raw)
    expect(result.map(item => item.title)).toEqual(['公开文章', '参考资料'])
    expect(result[0]).toMatchObject({ url: '/llms/rag/long', readingTime: 2, sourceCount: 0 })
    expect(result[1]).toMatchObject({ url: '/llms/rag/references/', readingTime: 1, sourceCount: 2 })
    expect(() => transform([
      article('/llms/rag/invalid.html', { ...publicFrontmatter, reviewed: undefined }, 'invalid public metadata')
    ])).toThrow('/llms/rag/invalid.html: reviewed is required for a published article')
  })
})
