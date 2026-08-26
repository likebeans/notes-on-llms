import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { checkBuiltSite } from '../../scripts/check-built-site'

const temporaryRoots: string[] = []

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(root => rm(root, { recursive: true, force: true })))
})

async function fixture(html: string, paths: Record<string, string> = {}): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), 'built-site-check-'))
  temporaryRoots.push(root)
  await writeFile(join(root, 'index.html'), html, 'utf8')
  await writeFile(join(root, 'sitemap.xml'), '<urlset></urlset>', 'utf8')

  await Promise.all(Object.entries(paths).map(async ([path, contents]) => {
    const target = join(root, path)
    await mkdir(join(target, '..'), { recursive: true })
    await writeFile(target, contents, 'utf8')
  }))

  return root
}

const validPage = `<!doctype html><html><head>
  <link rel="canonical" href="https://likebeans.github.io/notes-on-llms/">
  <link rel="icon" href="/notes-on-llms/logo.svg">
  <meta property="og:title" content="Notes on LLMs">
  <meta property="og:description" content="学习手册">
  <meta property="og:url" content="https://likebeans.github.io/notes-on-llms/">
  <meta property="og:image" content="https://likebeans.github.io/notes-on-llms/og-default.png">
  <meta name="twitter:card" content="summary_large_image">
</head><body>Page</body></html>`

describe('checkBuiltSite', () => {
  it('rejects a missing canonical', async () => {
    const root = await fixture(validPage.replace(/<link rel="canonical"[^>]+>\n?/, ''), {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
    })

    await expect(checkBuiltSite(root)).rejects.toThrow('canonical')
  })

  it('rejects canonicals that bypass the GitHub Pages base path', async () => {
    const root = await fixture(validPage.replace('/notes-on-llms/', '/'), {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
    })

    await expect(checkBuiltSite(root)).rejects.toThrow('canonical')
  })

  it('rejects Open Graph metadata using name instead of property', async () => {
    const root = await fixture(validPage.replace('property="og:title"', 'name="og:title"'), {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
    })

    await expect(checkBuiltSite(root)).rejects.toThrow('og:title')
  })

  it('rejects a favicon whose built target is missing', async () => {
    const root = await fixture(validPage, {
      'og-default.png': 'image',
    })

    await expect(checkBuiltSite(root)).rejects.toThrow('favicon')
  })

  it.each(['_drafts', 'superpowers', '开发计划'])('rejects forbidden built output paths containing %s', async forbidden => {
    const root = await fixture(validPage, {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
      [`${forbidden}/index.html`]: validPage,
    })

    await expect(checkBuiltSite(root)).rejects.toThrow(forbidden)
  })
})
