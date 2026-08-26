import { describe, expect, it } from 'vitest'
import { buildPageHead } from '../../docs/.vitepress/config/head'

describe('buildPageHead', () => {
  it('creates base-aware share metadata and Article JSON-LD for an article', () => {
    const head = buildPageHead({
      page: 'llms/rag/index.md',
      title: 'RAG 技术全景',
      description: 'RAG 学习入口',
      frontmatter: {
        pageType: 'article',
        author: 'likebeans',
        updated: '2026-08-25',
        reviewed: '2026-08-25',
      },
    })

    expect(head).toContainEqual([
      'link',
      { rel: 'canonical', href: 'https://likebeans.github.io/notes-on-llms/llms/rag/' },
    ])
    expect(head).toContainEqual(['meta', { property: 'og:title', content: 'RAG 技术全景' }])
    expect(head).toContainEqual(['meta', { name: 'twitter:card', content: 'summary_large_image' }])
    expect(JSON.stringify(head)).toContain('application/ld+json')
    const jsonLd = head.find(entry => entry[0] === 'script')?.[2]
    expect(JSON.parse(jsonLd as string)).toMatchObject({
      '@type': 'Article',
      mainEntityOfPage: 'https://likebeans.github.io/notes-on-llms/llms/rag/',
    })
  })

  it('does not create Article JSON-LD for non-article pages', () => {
    const head = buildPageHead({
      page: 'guide/index.md',
      title: '学习路径',
      description: '从基础到实践的学习路线',
      frontmatter: { pageType: 'path', updated: '2026-08-25' },
    })

    expect(head).toContainEqual([
      'link',
      { rel: 'canonical', href: 'https://likebeans.github.io/notes-on-llms/guide/' },
    ])
    expect(JSON.stringify(head)).not.toContain('application/ld+json')
  })

  it('creates the root canonical for the homepage', () => {
    const head = buildPageHead({
      page: 'index.md',
      title: 'Notes on LLMs',
      description: '大模型研究手册',
      frontmatter: { pageType: 'landing', updated: '2026-08-26' },
    })

    expect(head).toContainEqual([
      'link',
      { rel: 'canonical', href: 'https://likebeans.github.io/notes-on-llms/' },
    ])
  })

  it('keeps .html in canonicals for non-index pages when cleanUrls is disabled', () => {
    const head = buildPageHead({
      page: 'llms/rag/retrieval.md',
      title: '检索策略',
      description: 'RAG 检索策略',
      frontmatter: {
        pageType: 'article',
        updated: '2026-08-25',
        reviewed: '2026-08-25',
      },
    })

    expect(head).toContainEqual([
      'link',
      { rel: 'canonical', href: 'https://likebeans.github.io/notes-on-llms/llms/rag/retrieval.html' },
    ])
  })
})
