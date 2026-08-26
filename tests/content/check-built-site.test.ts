import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { checkBuiltSite } from '../../scripts/check-built-site'

const temporaryRoots: string[] = []

afterEach(async () => {
  vi.restoreAllMocks()
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

  it('rejects external canonicals that merely contain the GitHub Pages base path', async () => {
    const root = await fixture(
      validPage.replace(
        'href="https://likebeans.github.io/notes-on-llms/"',
        'href="https://example.com/notes-on-llms/"',
      ),
      {
        'logo.svg': '<svg/>',
        'og-default.png': 'image',
      },
    )

    await expect(checkBuiltSite(root)).rejects.toThrow('canonical')
  })

  it('rejects canonicals that only mention the GitHub Pages base path in a query string', async () => {
    const root = await fixture(
      validPage.replace(
        'href="https://likebeans.github.io/notes-on-llms/"',
        'href="https://likebeans.github.io/?next=/notes-on-llms/"',
      ),
      {
        'logo.svg': '<svg/>',
        'og-default.png': 'image',
      },
    )

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

  it('rejects every declared favicon target, not only the first one', async () => {
    const root = await fixture(
      validPage.replace(
        '<link rel="icon" href="/notes-on-llms/logo.svg">',
        '<link rel="icon" href="/notes-on-llms/logo.svg"><link rel="icon" href="/notes-on-llms/missing-icon.svg">',
      ),
      {
        'logo.svg': '<svg/>',
        'og-default.png': 'image',
      },
    )

    await expect(checkBuiltSite(root)).rejects.toThrow('missing-icon.svg')
  })

  it('rejects favicon targets that resolve to directories instead of files', async () => {
    const root = await fixture(validPage, {
      'og-default.png': 'image',
    })
    await mkdir(join(root, 'logo.svg'))

    await expect(checkBuiltSite(root)).rejects.toThrow('favicon')
  })

  it('rejects every declared share image target, not only the first one', async () => {
    const root = await fixture(
      validPage.replace(
        '<meta property="og:image" content="https://likebeans.github.io/notes-on-llms/og-default.png">',
        '<meta property="og:image" content="https://likebeans.github.io/notes-on-llms/og-default.png"><meta property="og:image" content="https://likebeans.github.io/notes-on-llms/missing-og.png">',
      ),
      {
        'logo.svg': '<svg/>',
        'og-default.png': 'image',
      },
    )

    await expect(checkBuiltSite(root)).rejects.toThrow('missing-og.png')
  })

  it.each(['_drafts', 'superpowers', '开发计划'])('rejects forbidden built output paths containing %s', async forbidden => {
    const root = await fixture(validPage, {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
      [`${forbidden}/index.html`]: validPage,
    })

    await expect(checkBuiltSite(root)).rejects.toThrow(forbidden)
  })

  it('reports only the HTML pages that were actually inspected', async () => {
    const log = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    const root = await fixture(validPage, {
      '404.html': '<!doctype html><html><head></head><body>Missing</body></html>',
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
    })

    await checkBuiltSite(root)

    expect(log).toHaveBeenCalledWith('Checked 1 non-404 built HTML pages.')
  })

  it('does not abort aggregation on malformed percent escapes in built paths', async () => {
    const root = await fixture(validPage, {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
      '%E0%A4%A/index.html': validPage,
    })

    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })
})
