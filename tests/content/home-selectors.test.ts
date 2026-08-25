import { describe, expect, it } from 'vitest'
import type { ContentIndexItem } from '../../docs/.vitepress/content/model'
import { selectRecentUpdates } from '../../docs/.vitepress/content/selectors'

const page = (overrides: Partial<ContentIndexItem>): ContentIndexItem => ({
  title: '文章',
  description: '测试文章',
  pageType: 'article',
  module: 'rag',
  level: 'intermediate',
  prerequisites: [],
  updated: '2026-08-25',
  reviewed: '2026-08-25',
  contentStatus: 'verified',
  techVersion: '2026',
  tags: ['test'],
  url: '/llms/rag/article',
  readingTime: 1,
  sourceCount: 0,
  ...overrides,
})

describe('selectRecentUpdates', () => {
  it('orders equal-date homepage updates by title', () => {
    const tiedPages = [
      page({ title: 'RAG', url: '/llms/rag/' }),
      page({ title: 'Agent', url: '/llms/agent/', module: 'agent' }),
    ]

    expect(selectRecentUpdates(tiedPages, 4).map(item => item.title)).toEqual(['Agent', 'RAG'])
  })
})
