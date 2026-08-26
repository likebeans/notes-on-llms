import { readdir, readFile, stat } from 'node:fs/promises'
import { relative, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { SITE_BASE, SITE_ORIGIN } from '../docs/.vitepress/config/site'

const FORBIDDEN_PATH_SEGMENTS = ['_drafts', 'superpowers', '开发计划']
const REQUIRED_OPEN_GRAPH = ['og:title', 'og:description', 'og:url', 'og:image']

type Attributes = Record<string, string>

function parseAttributes(tag: string): Attributes {
  const attributes: Attributes = {}
  for (const match of tag.matchAll(/([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g)) {
    const [, name, doubleQuoted, singleQuoted, unquoted] = match
    if (name.toLowerCase() === 'meta' || name.toLowerCase() === 'link') continue
    attributes[name.toLowerCase()] = doubleQuoted ?? singleQuoted ?? unquoted ?? ''
  }
  return attributes
}

function tags(source: string, name: string): string[] {
  return source.match(new RegExp(`<${name}\\b[^>]*>`, 'gi')) ?? []
}

function hasRel(attributes: Attributes, rel: string): boolean {
  return (attributes.rel ?? '').split(/\s+/).includes(rel)
}

function hasForbiddenPath(value: string): string | undefined {
  let decoded: string
  try {
    decoded = decodeURIComponent(value)
  } catch {
    decoded = value
  }
  decoded = decoded.toLowerCase()
  return FORBIDDEN_PATH_SEGMENTS.find(segment => decoded.includes(segment.toLowerCase()))
}

async function walk(directory: string): Promise<string[]> {
  const files: string[] = []
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name)
    if (entry.isDirectory()) files.push(...await walk(path))
    else if (entry.isFile()) files.push(path)
  }
  return files
}

async function assetExists(dist: string, reference: string): Promise<boolean> {
  let url: URL
  try {
    url = new URL(reference, SITE_ORIGIN)
  } catch {
    return false
  }

  if (url.origin !== SITE_ORIGIN || !url.pathname.startsWith(SITE_BASE)) return false
  const assetPath = url.pathname.slice(SITE_BASE.length)
  if (!assetPath || assetPath.startsWith('../')) return false

  try {
    const asset = await stat(resolve(dist, assetPath))
    return asset.isFile()
  } catch {
    return false
  }
}

function isCanonicalForSite(reference: string): boolean {
  try {
    const url = new URL(reference)
    return url.origin === SITE_ORIGIN && url.pathname.startsWith(SITE_BASE)
  } catch {
    return false
  }
}

function validatePageHead(path: string, source: string, errors: string[]): void {
  const links = tags(source, 'link').map(parseAttributes)
  const metas = tags(source, 'meta').map(parseAttributes)
  const canonical = links.find(attributes => hasRel(attributes, 'canonical'))

  if (!canonical?.href) errors.push(`${path}: missing canonical link`)
  else if (!isCanonicalForSite(canonical.href)) errors.push(`${path}: canonical must be under ${SITE_ORIGIN}${SITE_BASE}`)

  for (const property of REQUIRED_OPEN_GRAPH) {
    const meta = metas.find(attributes => attributes.property === property)
    if (!meta?.content) {
      const namedMeta = metas.find(attributes => attributes.name === property)
      errors.push(namedMeta
        ? `${path}: ${property} must use the property attribute`
        : `${path}: missing ${property} metadata`)
    }
  }

  if (!metas.some(attributes => attributes.name === 'twitter:card' && attributes.content)) {
    errors.push(`${path}: missing twitter:card metadata`)
  }
}

export async function checkBuiltSite(dist: string): Promise<void> {
  const root = resolve(dist)
  const files = await walk(root)
  const htmlFiles = files.filter(file => file.endsWith('.html')).sort()
  const inspectedHtmlFiles = htmlFiles.filter(file => relative(root, file).replaceAll('\\', '/') !== '404.html')
  const errors: string[] = []

  for (const file of files) {
    const path = relative(root, file).replaceAll('\\', '/')
    const forbidden = hasForbiddenPath(path)
    if (forbidden) errors.push(`${path}: built output must not contain ${forbidden}`)
  }

  for (const file of inspectedHtmlFiles) {
    const path = relative(root, file).replaceAll('\\', '/')

    const source = await readFile(file, 'utf8')
    validatePageHead(path, source, errors)

    const links = tags(source, 'link').map(parseAttributes)
    const favicons = links.filter(attributes => hasRel(attributes, 'icon'))
    if (!favicons.length) {
      errors.push(`${path}: missing favicon link`)
    }
    for (const favicon of favicons) {
      if (!favicon.href) errors.push(`${path}: missing favicon link`)
      else if (!await assetExists(root, favicon.href)) errors.push(`${path}: favicon target is missing or bypasses ${SITE_BASE}: ${favicon.href}`)
    }

    const shareImages = tags(source, 'meta')
      .map(parseAttributes)
      .filter(attributes => attributes.property === 'og:image')
    for (const shareImage of shareImages) {
      if (!shareImage.content) errors.push(`${path}: missing og:image metadata`)
      else if (!await assetExists(root, shareImage.content)) {
        errors.push(`${path}: share image target is missing or bypasses ${SITE_BASE}: ${shareImage.content}`)
      }
    }
  }

  const sitemap = resolve(root, 'sitemap.xml')
  try {
    const source = await readFile(sitemap, 'utf8')
    const forbidden = hasForbiddenPath(source)
    if (forbidden) errors.push(`sitemap.xml: must not contain ${forbidden}`)
  } catch {
    errors.push('missing sitemap.xml')
  }

  console.log(`Checked ${inspectedHtmlFiles.length} non-404 built HTML pages.`)
  if (errors.length) throw new Error(errors.join('\n'))
}

async function main(): Promise<void> {
  await checkBuiltSite(resolve('docs/.vitepress/dist'))
}

const invokedPath = process.argv[1] && pathToFileURL(resolve(process.argv[1])).href
if (invokedPath === import.meta.url) {
  main().catch(error => {
    console.error((error as Error).message)
    process.exitCode = 1
  })
}
