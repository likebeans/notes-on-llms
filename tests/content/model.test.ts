import { describe, expect, it } from 'vitest'
import { assertValidFrontmatter, isDraft } from '../../docs/.vitepress/content/model'

const article = {
  title: 'RAG 检索策略', description: '混合检索与重排序', pageType: 'article', module: 'rag',
  level: 'intermediate', prerequisites: ['/llms/rag/embedding'], updated: '2026-08-25',
  reviewed: '2026-08-25', contentStatus: 'verified', techVersion: '2025–2026', tags: ['rag']
}

describe('assertValidFrontmatter', () => {
  it('accepts a complete article', () => {
    expect(() => assertValidFrontmatter(article, 'llms/rag/retrieval.md')).not.toThrow()
  })
  it('rejects an article without reviewed', () => {
    expect(() => assertValidFrontmatter({ ...article, reviewed: undefined }, 'llms/rag/retrieval.md'))
      .toThrow('llms/rag/retrieval.md: reviewed is required for a published article')
  })
  it('allows landing pages without article-only fields', () => {
    expect(() => assertValidFrontmatter({
      title: '关于本站', description: '内容原则', pageType: 'landing', module: 'site',
      updated: '2026-08-25', contentStatus: 'verified', tags: ['about']
    }, 'about/index.md')).not.toThrow()
  })
})

it('identifies drafts', () => expect(isDraft({ contentStatus: 'draft' })).toBe(true))
