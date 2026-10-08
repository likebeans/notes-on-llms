import MarkdownIt from 'markdown-it'
import { SITE_BASE, SITE_ORIGIN } from '../docs/.vitepress/config/site'

type Reference = { kind: 'link' | 'image'; value: string }
type PageReferences = { anchors: Set<string>; references: Reference[] }

const markdownUtils = new MarkdownIt().utils
const VOID_TAGS = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'])
const RAW_TEXT_TAGS = new Set(['script', 'style', 'textarea', 'title', 'xmp', 'iframe', 'noembed', 'noframes', 'noscript'])

function decodeAttribute(value: string): string {
  // Reuse the existing Markdown dependency's entity table without its Markdown backslash unescaping.
  return markdownUtils.unescapeAll(value.replaceAll('\\', '&#92;'))
}

function attributes(tag: string): Record<string, string> {
  const result: Record<string, string> = {}
  const body = tag.replace(/^<\/?[^\s/>]+/, '').replace(/\/?>$/, '')
  for (const match of body.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)) {
    const name = match[1].toLowerCase()
    // Browsers retain the first occurrence of an attribute.
    if (!(name in result)) result[name] = decodeAttribute(match[2] ?? match[3] ?? match[4] ?? '')
  }
  return result
}

function collectPage(source: string): PageReferences {
  const anchors = new Set<string>()
  const references: Reference[] = []
  const stack: { tag: string; inert: boolean; mermaid: boolean; svg: boolean }[] = []
  let rawTextTag: string | undefined
  // Tokenize generated HTML, respecting quoted > characters, comments and raw-text elements.
  // This is not a general HTML error-recovery parser; VitePress produces the input markup.
  const tokens = /<!--[\s\S]*?(?:-->|$)|<![^>]*>|<\/?([a-zA-Z][\w:-]*)\b(?:[^"'<>]|"[^"]*"|'[^']*')*>/g
  for (const match of source.matchAll(tokens)) {
    if (!match[1]) continue
    const tag = match[1].toLowerCase()
    const closing = match[0].startsWith('</')
    if (rawTextTag) {
      if (closing && tag === rawTextTag) rawTextTag = undefined
      continue
    }
    if (closing) {
      const index = stack.map(frame => frame.tag).lastIndexOf(tag)
      if (index >= 0) stack.splice(index)
      continue
    }
    const attrs = attributes(match[0])
    const parent = stack[stack.length - 1]
    const inert = parent?.inert ?? false
    const mermaid = (parent?.mermaid ?? false) || (attrs.class ?? '').split(/\s+/).includes('mermaid')
    const svg = (parent?.svg ?? false) || tag === 'svg'
    if (!inert) {
      if (attrs.id) anchors.add(attrs.id)
      if (tag === 'a' && attrs.name) anchors.add(attrs.name)
      if (tag === 'a' && 'href' in attrs) {
        // Mermaid may create these SVG IDs only after hydration. Ordinary article anchors remain checked.
        if (!(mermaid && svg && attrs.href.startsWith('#'))) references.push({ kind: 'link', value: attrs.href })
      }
      if (tag === 'img' && 'src' in attrs) references.push({ kind: 'image', value: attrs.src })
    }
    if (tag === 'plaintext') break
    if (RAW_TEXT_TAGS.has(tag)) rawTextTag = tag
    else if (!VOID_TAGS.has(tag) && !match[0].endsWith('/>')) {
      stack.push({ tag, inert: inert || tag === 'template', mermaid, svg })
    }
  }
  return { anchors, references }
}

function documentUrl(path: string): URL {
  const encoded = path.split('/').map(encodeURIComponent).join('/')
  // A directory index is visited at its directory URL, so relative links inherit that directory.
  const publicPath = encoded === 'index.html' ? '' : encoded.replace(/\/index\.html$/, '/')
  return new URL(`${SITE_BASE}${publicPath}`, SITE_ORIGIN)
}

function decodedLocalPath(url: URL): string {
  const baseWithoutSlash = SITE_BASE.replace(/\/$/, '')
  if (url.pathname === baseWithoutSlash) return ''
  if (!url.pathname.startsWith(SITE_BASE)) throw new Error(`URL bypasses ${SITE_BASE}`)
  const segments = url.pathname.slice(SITE_BASE.length).split('/').map(segment => decodeURIComponent(segment))
  if (segments.some(segment => /[/\\\0]/.test(segment) || segment === '.' || segment === '..')) {
    throw new Error('encoded path separator or path escape')
  }
  return segments.join('/')
}

function pageAliases(paths: Iterable<string>): Map<string, string> {
  const aliases = new Map<string, string>()
  for (const path of [...paths].sort()) {
    const keys = [path, path.replace(/\.html$/, '')]
    if (path === 'index.html') keys.push('')
    else if (path.endsWith('/index.html')) keys.push(path.slice(0, -10), path.slice(0, -11))
    for (const key of keys) if (!aliases.has(key)) aliases.set(key, path)
  }
  return aliases
}

export function validateBuiltReferences(sources: ReadonlyMap<string, string>, files: ReadonlySet<string>): string[] {
  const pages = new Map([...sources].map(([path, source]) => [path, collectPage(source)]))
  const aliases = pageAliases(pages.keys())
  const errors: string[] = []
  for (const [path, page] of pages) {
    for (const reference of page.references) {
      const { kind, value } = reference
      if (kind === 'image' && !value.trim()) {
        errors.push(`${path}: missing image target: ${value}`)
        continue
      }
      let url: URL
      let localPath: string
      try {
        url = new URL(value, documentUrl(path))
        if (!['http:', 'https:'].includes(url.protocol) || url.origin !== SITE_ORIGIN) continue
        localPath = decodedLocalPath(url)
      } catch (error) {
        errors.push(`${path}: invalid ${kind} target: ${value} (${(error as Error).message})`)
        continue
      }
      const target = files.has(localPath) ? localPath : kind === 'link' ? aliases.get(localPath) : undefined
      if (!target) {
        errors.push(`${path}: missing ${kind} target: ${value}`)
        continue
      }
      if (kind === 'image' || !url.hash || !pages.has(target)) continue
      const anchors = pages.get(target)!.anchors
      const fragment = url.hash.slice(1)
      // HTML fragment lookup checks the literal ID first, then one percent-decoding pass.
      if (anchors.has(fragment)) continue
      try {
        const decoded = decodeURIComponent(fragment)
        if (!decoded || decoded.toLowerCase() === 'top' || anchors.has(decoded)) continue
      } catch {
        errors.push(`${path}: malformed anchor encoding: ${value}`)
        continue
      }
      errors.push(`${path}: missing anchor: ${value} (target ${target})`)
    }
  }
  return errors
}
