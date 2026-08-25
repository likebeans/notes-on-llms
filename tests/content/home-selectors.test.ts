import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
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

describe('homepage static contracts', () => {
  it('targets the VitePress-generated roadmap anchors for both long learning paths', () => {
    const learningPaths = readFileSync(new URL('../../docs/.vitepress/theme/components/home/LearningPaths.vue', import.meta.url), 'utf8')

    expect(learningPaths).toContain("href: '/guide/roadmap#路径一-应用开发者-3-6-个月'")
    expect(learningPaths).toContain("href: '/guide/roadmap#路径二-算法工程师-6-12-个月'")
  })

  it('removes the VitePress width limit only for pages containing the custom homepage', () => {
    const homeStyles = readFileSync(new URL('../../docs/.vitepress/theme/styles/home.css', import.meta.url), 'utf8')

    expect(homeStyles).toContain('.VPPage:has(.nl-home-page), .VPDoc:has(.nl-home-page) .container, .VPDoc:has(.nl-home-page) .content, .VPDoc:has(.nl-home-page) .content-container { max-width: none; width: 100%; }')
  })
})
