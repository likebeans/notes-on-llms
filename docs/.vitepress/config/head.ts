import type { HeadConfig } from 'vitepress'
import type { ContentFrontmatter } from '../content/model'
import { SITE_AUTHOR, SITE_ORIGIN, SITE_URL } from './site'

type PageHeadContext = {
  page: string
  title: string
  description: string
  frontmatter: Partial<ContentFrontmatter>
}

const pageUrl = (page: string): string => {
  const path = page.replace(/^\/+/, '').replace(/\.md$/, '')
  const route = path === 'index' ? '' : path.replace(/\/index$/, '')
  return `${SITE_URL}${route}${route && !route.endsWith('/') ? (page.endsWith('/index.md') ? '/' : '') : ''}`
}

export function buildPageHead({ page, title, description, frontmatter }: PageHeadContext) {
  const url = pageUrl(page)
  const image = `${SITE_URL}logo.png`
  const head: HeadConfig[] = [
    ['link', { rel: 'canonical', href: url }],
    ['meta', { property: 'og:type', content: frontmatter.pageType === 'article' ? 'article' : 'website' }],
    ['meta', { property: 'og:title', content: title }],
    ['meta', { property: 'og:description', content: description }],
    ['meta', { property: 'og:url', content: url }],
    ['meta', { property: 'og:image', content: image }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: title }],
    ['meta', { name: 'twitter:description', content: description }],
  ]

  if (frontmatter.pageType === 'article') {
    head.push(['script', { type: 'application/ld+json' }, JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'Article',
      headline: title,
      author: { '@type': 'Person', name: frontmatter.author ?? SITE_AUTHOR },
      dateModified: frontmatter.updated,
      lastReviewed: frontmatter.reviewed,
      url,
      mainEntityOfPage: url,
      publisher: { '@type': 'Organization', name: 'Notes on LLMs', url: SITE_ORIGIN },
    })])
  }

  return head
}
