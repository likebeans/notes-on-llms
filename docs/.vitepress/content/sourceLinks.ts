import MarkdownIt from 'markdown-it'

// Parse links without initializing VitePress's shared renderer or highlighter.
// Keep its HTML/linkification rules while excluding code and image tokens.
const md = new MarkdownIt({ html: true, linkify: true })
md.linkify.set({ fuzzyLink: false })

export async function extractSourceLinks(source: string): Promise<string[]> {
  const links = new Set<string>()
  const add = (href: string) => {
    try {
      const url = new URL(href.replace(/&amp;/g, '&'))
      if (!['https:', 'http:'].includes(url.protocol)) return
      url.hash = ''
      links.add(url.href)
    } catch { /* Relative links are navigation, not external sources. */ }
  }
  type Token = { type: string; content: string; children?: Token[] | null; attrGet(name: string): string | null }
  const visit = (tokens: Token[]) => {
    for (const token of tokens) {
      if (token.type === 'link_open') add(token.attrGet('href') ?? '')
      if (token.type === 'html_inline' || token.type === 'html_block') {
        for (const match of token.content.matchAll(/<a\b[^>]*\bhref\s*=\s*['"]([^'"]+)['"]/gi)) add(match[1])
        for (const block of token.content.matchAll(/<SourceList\b[\s\S]*?(?:\/>|<\/SourceList>)/g)) {
          for (const match of block[0].matchAll(/\bhref:\s*['"]([^'"]+)['"]/g)) add(match[1])
        }
      }
      if (token.type !== 'image' && token.children) visit(token.children)
    }
  }
  visit(md.parse(source.replace(/^---[\s\S]*?---\s*/, ''), {}))
  return [...links]
}
