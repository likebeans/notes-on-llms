import { createContentLoader } from 'vitepress'
import { assertValidFrontmatter, isDraft, type ContentIndexItem } from './model'
import { normalizeUrl } from './selectors'
import { findModuleOrder } from '../config/modules'
import { extractSourceLinks } from './sourceLinks'

declare const data: ContentIndexItem[]
export { data }

export default createContentLoader('**/*.md', {
  includeSrc: true,
  async transform(raw): Promise<ContentIndexItem[]> {
    const pages = await Promise.all(raw
      .filter(page => {
        const url = normalizeUrl(page.url)
        return url !== '/superpowers' && !url.startsWith('/superpowers/') && url !== '/开发计划'
      })
      .map(async page => {
        assertValidFrontmatter(page.frontmatter, page.url)
        const frontmatter = page.frontmatter
        const source = page.src ?? ''
        const body = source.replace(/^---[\s\S]*?---/, '')
        const words = body.split(/\s+|(?=[\u4e00-\u9fff])/).filter(Boolean).length
        const url = page.url.replace(/\.html$/, '').replace(/\/index$/, '/')
        return {
          ...frontmatter, url,
          order: frontmatter.order ?? findModuleOrder(frontmatter.module, url),
          readingTime: Math.max(1, Math.ceil(words / 350)),
          sourceCount: (await extractSourceLinks(source)).length,
        } as ContentIndexItem
      }))
    return pages.filter(page => !isDraft(page))
  },
})
