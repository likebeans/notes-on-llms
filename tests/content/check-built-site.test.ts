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

  it('rejects canonicals that do not exactly match the built output path', async () => {
    const root = await fixture(validPage, {
      'llms/rag/retrieval.html': validPage.replace(
        'href="https://likebeans.github.io/notes-on-llms/"',
        'href="https://likebeans.github.io/notes-on-llms/llms/rag/retrieval"',
      ),
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
    })

    await expect(checkBuiltSite(root)).rejects.toThrow('canonical must match')
  })

  it('does not abort aggregation on malformed percent escapes in built paths', async () => {
    const encodedPathPage = validPage.replace(
      'href="https://likebeans.github.io/notes-on-llms/"',
      'href="https://likebeans.github.io/notes-on-llms/%E0%A4%A/"',
    )
    const root = await fixture(validPage, {
      'logo.svg': '<svg/>',
      'og-default.png': 'image',
      '%E0%A4%A/index.html': encodedPathPage,
    })

    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })
})

const publicRoot = 'https://likebeans.github.io/notes-on-llms/'
const sharedAssets = { 'logo.svg': '<svg/>', 'og-default.png': 'image' }

function pageAt(path: string, body: string): string {
  return validPage
    .replace(`href="${publicRoot}"`, `href="${publicRoot}${path}"`)
    .replace('<body>Page</body>', `<body>${body}</body>`)
}

describe('built internal references', () => {
  it('reports the source page and original href when a page is missing', async () => {
    const root = await fixture(pageAt('', '<a href="/notes-on-llms/missing?x=1#part">Broken</a>'), sharedAssets)
    await expect(checkBuiltSite(root)).rejects.toThrow('index.html: missing link target: /notes-on-llms/missing?x=1#part')
  })

  it('resolves relative links against the current page and supports page aliases', async () => {
    const root = await fixture(pageAt('', '<a href="guide/">Guide</a><a href="guide/index.html">Index</a><a href="guide/index">Index alias</a>'), {
      ...sharedAssets,
      'guide/index.html': pageAt('guide/', '<a href="./topic?mode=read#section">Topic</a>'),
      'guide/topic.html': pageAt('guide/topic.html', '<h2 id="section">Topic</h2><a href="../">Home</a><a href="./topic.html#section">Self</a>'),
    })
    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })

  it('matches actual rendered IDs, including the underscore for numeric headings', async () => {
    const root = await fixture(pageAt('', '<a href="article#_1-背景">Good</a><a href="article#1-背景">Bad</a>'), {
      ...sharedAssets,
      'article.html': pageAt('article.html', '<h2 id="_1-背景">1. 背景</h2>'),
    })
    await expect(checkBuiltSite(root)).rejects.toThrow('index.html: missing anchor: article#1-背景')
  })

  it('decodes encoded Chinese paths, fragments and HTML attribute entities once', async () => {
    const root = await fixture(pageAt('', '<a href="%E4%B8%AD%E6%96%87.html?q=a&amp;b=c#%E7%AB%A0%E8%8A%82">中文</a><a href="#a&amp;b">Entity</a><a href="#a+b">Plus</a><a href="#%2520">Literal percent</a><h2 id="a&amp;b">Entity</h2><i id="a+b"></i><i id="%20"></i>'), {
      ...sharedAssets,
      '中文.html': pageAt('中文.html', '<h2 id="章节">章节</h2>'),
    })
    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })

  it('allows same-page queries, empty fragments, top and legacy named anchors', async () => {
    const root = await fixture(pageAt('', '<h2 id="section"></h2><a name="legacy"></a><a href="?print=1#section">Query</a><a href="#">Top</a><a href="#TOP">Top</a><a href="">Self</a><a href="#legacy">Legacy</a>'), sharedAssets)
    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })

  it('checks same-origin absolute URLs while skipping external and non-HTTP links', async () => {
    const root = await fixture(pageAt('', '<a href="https://example.com/missing#gone">External</a><a href="//cdn.example.com/file">CDN</a><a href="mailto:reader@example.com">Mail</a><a href="https://likebeans.github.io/notes-on-llms/missing">Internal</a>'), sharedAssets)
    await expect(checkBuiltSite(root)).rejects.toThrow('missing link target: https://likebeans.github.io/notes-on-llms/missing')
  })

  it.each(['/outside.html', '../outside.html', '/notes-on-llms-other/page', '/notes-on-llms/%2e%2e/outside.html', '/notes-on-llms/folder%2f..%2foutside.html'])('rejects base bypass or encoded path escape: %s', async href => {
    const root = await fixture(pageAt('', `<a href="${href}">Escape</a>`), sharedAssets)
    await expect(checkBuiltSite(root)).rejects.toThrow(href)
  })

  it('does not accept a trailing slash by falling back to a sibling .html file', async () => {
    const root = await fixture(pageAt('', '<a href="article/">Wrong route</a>'), {
      ...sharedAssets, 'article.html': pageAt('article.html', '<h1>Article</h1>'),
    })
    await expect(checkBuiltSite(root)).rejects.toThrow('missing link target: article/')
  })

  it('checks local image files with relative paths, query strings and encoded names', async () => {
    const root = await fixture(pageAt('', '<a href="guide/page.html">Read</a>'), {
      ...sharedAssets,
      'guide/page.html': pageAt('guide/page.html', '<img src="../assets/%E5%9B%BE%20%E7%89%87.svg?v=2#icon"><img src="data:image/svg+xml,%3Csvg%3E"><img src="https://example.com/missing.png">'),
      'assets/图 片.svg': '<svg/>',
    })
    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })

  it.each(['missing.png', '', '/missing.png'])('reports broken local image src: %s', async src => {
    const root = await fixture(pageAt('', `<img src="${src}">`), sharedAssets)
    await expect(checkBuiltSite(root)).rejects.toThrow('index.html:')
  })

  it('does not accept a directory as an image file', async () => {
    const root = await fixture(pageAt('', '<img src="images/">'), sharedAssets)
    await mkdir(join(root, 'images'))
    await expect(checkBuiltSite(root)).rejects.toThrow('image')
  })

  it('does not treat comments, scripts or inert templates as rendered links or anchors', async () => {
    const root = await fixture(pageAt('', '<!-- <a href="missing"> --><script>const example = \'<a href="missing">\';</script><template><a href="missing">Hidden</a><i id="hidden"></i></template><a href="#hidden">Not rendered</a>'), sharedAssets)
    let message = ''
    try { await checkBuiltSite(root) } catch (error) { message = (error as Error).message }
    expect(message).toContain('missing anchor: #hidden')
    expect(message).not.toContain('missing link target: missing')
  })

  it('does not inspect SVG marker references or dynamically generated Mermaid fragment links', async () => {
    const root = await fixture(pageAt('', '<div class="mermaid"><svg><use href="#runtime-marker"></use><a href="#runtime-node">Generated</a></svg></div><h2 id="heading">Heading</h2><a href="#heading">Normal</a>'), sharedAssets)
    await expect(checkBuiltSite(root)).resolves.toBeUndefined()
  })

  it('still rejects broken ordinary anchors on pages that contain Mermaid', async () => {
    const root = await fixture(pageAt('', '<div class="mermaid"></div><a href="#missing-heading">Broken</a>'), sharedAssets)
    await expect(checkBuiltSite(root)).rejects.toThrow('missing anchor: #missing-heading')
  })

  it('checks links in 404 pages without requiring their page head', async () => {
    const root = await fixture(validPage, { ...sharedAssets, '404.html': '<html><body><a href="/notes-on-llms/missing">Home?</a></body></html>' })
    await expect(checkBuiltSite(root)).rejects.toThrow('404.html: missing link target: /notes-on-llms/missing')
  })

  it('does not use 404.html as a fallback for a missing route', async () => {
    const root = await fixture(pageAt('', '<a href="missing">Missing</a>'), { ...sharedAssets, '404.html': '<html><body>404</body></html>' })
    await expect(checkBuiltSite(root)).rejects.toThrow('missing link target: missing')
  })

  it('aggregates malformed references with missing targets instead of aborting', async () => {
    const root = await fixture(pageAt('', '<a href="%E0%A4%A">Malformed</a><a href="missing">Missing</a><img src="missing.png">'), sharedAssets)
    let message = ''
    try { await checkBuiltSite(root) } catch (error) { message = (error as Error).message }
    expect(message).toContain('%E0%A4%A')
    expect(message).toContain('missing link target: missing')
    expect(message).toContain('missing image target: missing.png')
  })
})
