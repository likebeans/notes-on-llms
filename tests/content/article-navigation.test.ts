import { describe, expect, it } from 'vitest'
import type { ContentIndexItem } from '../../docs/.vitepress/content/model'
import { findAdjacentArticle } from '../../docs/.vitepress/content/selectors'

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
  techVersion: '2026-08',
  tags: ['rag'],
  url: '/llms/rag/article',
  readingTime: 3,
  sourceCount: 2,
  ...overrides,
})

const pages = [
  page({ title: 'RAG 概述', url: '/llms/rag/', order: 1 }),
  page({ title: '检索策略', url: '/llms/rag/retrieval', order: 2 }),
]

describe('article navigation', () => {
  it('normalizes route and candidate URLs before finding adjacent articles', () => {
    expect(findAdjacentArticle(pages, '/llms/rag')).toEqual(findAdjacentArticle(pages, '/llms/rag/'))
    expect(findAdjacentArticle(pages, '/llms/rag/').next?.url).toBe('/llms/rag/retrieval')
  })

  it('treats an unordered article as the final deterministic entry', () => {
    expect(findAdjacentArticle([{ ...pages[0], order: undefined }], '/llms/rag/').next).toBeUndefined()
  })

  it('does not create custom adjacent links for site-level auxiliary articles', () => {
    const sitePages = [
      page({ title: 'Agent 面试题', module: 'site', url: '/interviews/agent-questions', order: undefined }),
      page({ title: 'Checklist', module: 'site', url: '/reference/checklists', order: undefined }),
    ]

    expect(findAdjacentArticle(sitePages, '/interviews/agent-questions')).toEqual({})
  })
})
