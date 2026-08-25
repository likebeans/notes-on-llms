import { createContentLoader } from 'vitepress'
import { assertValidFrontmatter, isDraft, type ContentIndexItem } from './model'

declare const data: ContentIndexItem[]
export { data }

export default createContentLoader('**/*.md', {
  includeSrc: true,
  transform(raw): ContentIndexItem[] {
    return raw
      .filter(page => !page.url.startsWith('/superpowers/') && page.url !== '/开发计划')
      .map(page => {
        assertValidFrontmatter(page.frontmatter, page.url)
        const source = page.src ?? ''
        const body = source.replace(/^---[\s\S]*?---/, '')
        const words = body.split(/\s+|(?=[\u4e00-\u9fff])/).filter(Boolean).length
        const references = source.match(/^##\s+参考资料[\s\S]*$/m)?.[0] ?? ''

        return {
          ...page.frontmatter,
          url: page.url.replace(/\.html$/, '').replace(/\/index$/, '/'),
          readingTime: Math.max(1, Math.ceil(words / 350)),
          sourceCount: (references.match(/^\s*[-*]\s+/gm) ?? []).length
        } as ContentIndexItem
      })
      .filter(page => !isDraft(page))
  }
})
